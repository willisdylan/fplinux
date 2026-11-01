// SPDX-License-Identifier: GPL-2.0-only
/*
 * f286-radio.c - Ruio F286 (UMS9117/T107) baseband power-on and DSP bring-up.
 *
 * The register sequences here are the stock firmware's, read from its kernel
 * image (addresses below are stock kernel VAs unless they are registers).
 * docs/bringup-status.md in this repository says which parts have run on a
 * real phone and with what result.
 *  - radio_power: whale-die power-on chain (16-bit AON gate SETs with a latch
 *    check, keyed step-0, analog ADI batch, radio-on set, 3 s c4/d4 poll),
 *    then the WHu3g island arm (0x402b203c = 0x02000000 and a triple-stable
 *    c0[4:0] == 0 poll, the tail of the stock whale power-on).
 *  - init:       lazy maps (MMIO + DSP AP windows) and the GICD IRQ mask.
 *  - load_lte:   LTE DSP boot (0x2e4000 bytes of firmware -> 0x80448000) in
 *                the stock pre-boot order: start-ctrl low + reset hold, four
 *                PMU clears (incl. 0x402b2034), cores poll to 0, staging cell
 *                0x80447E00, ring format, clock (0x60831044 = 0x40 tail and
 *                0x402D02B4 mux = TWPLL_51M2), boot-addr 0x402e0248 =
 *                0x40224200, start-ctrl high, release. The stock cold boot
 *                sends the DSP no command; cmdId 6 is opt-in (send_cmd6).
 *                Ring sends: payloads up to 0x190 bytes go inline, bigger ones
 *                take the stock split path (marker 0xaaaa, u16 slot handle at
 *                body+8, 4 x 0x2800 slots in the second LTE map 0x8073b000).
 *  - l1 pump:    persistent kthread (f286-l1pump) that drains both 12-slot
 *                response rings every 5 ms (draining every 5 s overflowed the
 *                ring in DL bursts); type 5 = LTEDSP_Ver handshake, type 7 =
 *                DSP assert.
 *  - load_gsm:   GSM DSP boot (3 MB of firmware -> 0x80100000), param blocks
 *                at window 0x800e0000 + stock offsets (+0x19670 block, header
 *                word written last), boot-addr 0x402e0250 = 0x40080040
 *                (entry 0x80100080 >> 1), run word + rf_open/rf_close check.
 *                The release is 0x402b2038 = 0x03000000 + 0x402b2044 =
 *                0x03000000, then a poll for c0 & 0x7c00 == 0 (the stock
 *                xdsp_power_release_wait for DSP id 2).
 *  - status:     PMU power-domain snapshot.  queue: both DSP command areas.
 *  - window_gsm/window_lte: full 0x12000-byte AP-window dumps.
 *  - axidump, gatetest: one-shot diagnostics, see their comments.
 *
 * Lessons from hardware that shaped the code:
 *  - sysfs attr names must avoid the always-present device attrs (a `power`
 *    attr collides with the runtime-PM `power` directory and deadlocks
 *    kernfs; hence `radio_power`).
 *  - Probe creates only the `init` attr and schedules a 60 s work; the other
 *    attrs are created one at a time from the work with a log line before
 *    each, so any hang happens with console + SSH up.
 *  - AON gate SET registers are 16-bit (32-bit stores are silently dropped);
 *    the keyed 0x4031 block is 16-bit only; gates are verified latched
 *    before any keyed access (ungated stores bus-abort: external abort
 *    0x1008, kernfs active-ref leak, subtree locked until reboot).
 *  - The 0x515d/0x515e die-local pages external-aborted on direct probes
 *    because they are the wrong bus addresses: the stock MMU maps this range
 *    1:1, so the real physical addresses are 0xD15D/E0000 (checked at
 *    instruction level). The light-path misc page is 0xE0100000 (0x60101000
 *    was equally wrong). The clock-on sequence touches 0xD15E/D only inside
 *    a fire-and-forget kthread, so a death there cannot take the load
 *    thread down.
 *  - ADI writes take the ADI user-lock (0x40600224, read-zero semaphore,
 *    release 0x5348554C) so they cannot collide with spi-sprd-adi.
 *  - Unclaimed baseband IRQs are masked directly in the GIC distributor
 *    (ICENABLER2 0x12001188 <- 0xE0000030 = hwirq 68/69/93/94/95;
 *    ICENABLER3 0x1200118C <- 0xE0000000 = hwirq 125/126/127). TODO: real
 *    handlers, and settle the SPI numbering (see f286_radio_power_locked).
 *
 * Memory model: the DT memory node starts at 0x80760000, above the whole DSP
 * region; 0x80000000-0x80760000 is not RAM to Linux. (An earlier base of
 * 0x80740000 overlapped the LTE UL-data slots [0x80745000,0x8075E000).) The
 * DSP windows and firmware targets therefore map with memremap(MEMREMAP_WC)
 * and fall back to ioremap_wc. WC keeps stores uncached for the
 * non-coherent DSPs.
 */

#include <linux/cacheflush.h> /* clean_dcache_area (ARM-only platform) */
#include <linux/delay.h>
#include <linux/device.h>
#include <linux/err.h>
#include <linux/firmware.h>
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/io.h>
#include <linux/iopoll.h>
#include <linux/jiffies.h>
#include <linux/kthread.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/of.h>
#include <linux/platform_device.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/sysfs.h>
#include <linux/unaligned.h>
#include <linux/vmalloc.h>

/* Physical window map (hardcoded like the rest of the F286 port). */
#define F286_ADI_PHYS 0x40600000UL /* ADI controller + slave window */
#define F286_ADI_SPAN 0x00010000UL
#define F286_PMU_PHYS 0x402b0000UL /* PMU APB incl. SET/CLEAR banks */
#define F286_PMU_SPAN 0x00003000UL
#define F286_AON_PHYS 0x402e0000UL /* AON_APB + AON2 + AON3 banks */
#define F286_AON_SPAN 0x00003000UL
#define F286_KEY_PHYS 0x40310000UL /* keyed block, 16-bit ONLY */
#define F286_KEY_SPAN 0x00001000UL
#define F286_WHALE_APB_PHYS 0x60830000UL /* whale APB (power request) */
#define F286_WHALE_APB_SPAN 0x00003000UL /* incl. SET/CLEAR alias banks */
#define F286_CLKM_PHYS 0x402D0000UL /* AP clock mgr (XDSP mux 0x402D02B4) */
#define F286_CLKM_SPAN 0x00001000UL
#define F286_WCFG_PHYS \
	0xE0100000UL /* whale misc die cfg (1:1 mapped in stock) */
#define F286_WCFG_SPAN 0x00002000UL
#define F286_WAON_D_PHYS \
	0xD15D0000UL /* whale RF gate (identity; old 0x515d wrong bus) */
#define F286_WAON_D_SPAN 0x00001000UL
#define F286_WAON_E_PHYS \
	0xD15E0000UL /* whale AON/PMU (identity; old 0x515e wrong bus) */
#define F286_WAON_E_SPAN 0x00003000UL
#define F286_BLK404D_PHYS 0x404d0000UL
#define F286_BLK404D_SPAN 0x00001000UL
#define F286_XDSP_PHYS \
	0xE0400000UL /* LTE XDSP core A/B windows (identity, stock 0x80d2d5d0) */
#define F286_XDSP_SPAN 0x00100000UL
#define F286_XDSP_PARAM 0xf2370000 /* stock boot param -> win+0x40 */
#define F286_XDSP_IMG_MAX 0x10000 /* largest image is 0x9ab4 */
#define F286_GICD_PHYS 0x12001000UL
#define F286_GICD_SPAN 0x00001000UL
#define F286_GSM_WIN_PHYS 0x800e0000UL /* GSM DSP AP window */
#define F286_LTE_WIN_PHYS 0x80400000UL /* LTE DSP AP window */
#define F286_WIN_SPAN 0x00012000UL
#define F286_WIN_ALIVE_DUMP 0x10bf8 /* stock alive-check dump word (0x33) */
#define F286_WIN_SLEEP_SYNC 0x00e14 /* DSP writes 0x22222222 on resume */
#define F286_GSM_DRAM_PHYS 0x80100000UL /* GSM DSP image */
#define F286_GSM_DRAM_SPAN 0x00300000UL
#define F286_GSM_BASE_SPAN \
	0x00320000UL /* 0x800e0000 window base .. image end */
#define F286_GSM_PARAMS 0x19670 /* param block (win-rel; 0x800f9670) */
#define F286_GSM_RUN_WORD (F286_GSM_PARAMS + 0x634) /* 0x800f9ca4 */
#define F286_GSM_RF_OPEN (F286_GSM_PARAMS + 0x770) /* u16, live stock 0x0133 */
#define F286_GSM_RF_CLOSE (F286_GSM_PARAMS + 0x772) /* u16, live stock 0x0087 */
#define F286_GSM_AUDIO 0x2a8000 /* audio area 0x80388000 */
#define F286_GSM_IRAM_PHYS \
	0x00010000UL /* AP IRAM shared with the DSP [inferred] */
#define F286_GSM_IRAM_SPAN 0x00002000UL
#define F286_LTE_DRAM_PHYS 0x80448000UL /* LTE DSP image */
#define F286_LTE_DRAM_SPAN 0x002e4000UL
#define F286_HS_WIN_PHYS 0x80447000UL /* 2nd window base (handshake cells) */
#define F286_HS_WIN_SPAN 0x00001000UL
#define F286_HS_NV9 0x0e00 /* nv9|tick<<16 stage = AP 0x80447E00 */
#define F286_HS_AWAKE 0x0c40 /* stock awake word = AP 0x80447C40 */
#define F286_HS_SYNC 0x0e14 /* sleep-sync word = AP 0x80447E14 */

/* ADI controller registers (offsets from F286_ADI_PHYS). */
#define F286_ADI_RD_CMD 0x0028
#define F286_ADI_RD_DATA 0x002c
#define F286_ADI_ARM_FIFO_STS 0x0030
#define F286_ADI_USER_LOCK 0x0224
#define F286_ADI_FIFO_EMPTY BIT(10)
#define F286_ADI_FIFO_FULL BIT(11)
#define F286_ADI_RD_BUSY BIT(31)
#define F286_ADI_UNLOCK_TOKEN 0x5348554cUL

/* PMU / AON offsets. */
#define F286_PMU_POWER_STATUS 0x00c0 /* [4:0]=WHu3g [19:15]=CM4 */
#define F286_PMU_WTLCP_STATUS 0x00c4 /* [4:0] = WTLCP (LTE DSP) */
#define F286_PMU_WHALE_FSM 0x00d4 /* [7:4] = whale FSM, 6 = READY */
#define F286_PMU_RD_00CC 0x00cc
#define F286_PMU_RD_0044 0x0044
#define F286_PMU_CLR_DSP_CFG 0x20cc /* bank2 CLEAR alias of 0x00cc */
#define F286_PMU_CLR_2044 0x2044 /* bank2 CLEAR alias of 0x0044 */
#define F286_PMU_CLR_2034 \
	0x2034 /* bank2 CLEAR alias of 0x0034 (XDSP core req) */
#define F286_PMU_CLR_2038 \
	0x2038 /* bank2 CLEAR alias of 0x0038 (GSM core req) */
#define F286_PMU_CLR_203C 0x203c /* bank2 CLEAR alias of 0x003c (WHu3g arm) */
#define F286_PMU_RD_0034 0x0034
/* CP_SOFT_RST CLR alias (same write as stock): 0x402b00b0 resets to
 * 0x507 = WTLCP_SOFT_RST b0 + PUBCP_SOFT_RST b1 + WTLCP_DSP_SYS_SRST b2.
 * Stock clears bits 0+2 (write 5) in its GSM pre-boot
 * 0x80D0FA5E and OS power sweep 0x80D0FAAA. NOTE: same numeric offset as
 * F286_AON3_BRIDGE below, but a DIFFERENT block (this one is pmu 0x402b).
 */
#define F286_PMU_CLR_CPSOFT_RST 0x20b0
#define F286_AON_DSP_RESET 0x0130 /* bit16 LTE / bit17 GSM hold */
/* AON_REG_PROT: writing 0x9620 is the REG_PROT unlock key for the
 * protected LDSP-control register group (b31 LDSP_CTRL_PROT reads 1 only
 * when the key == 0x9620). Stock HAL 0x811d0db4 writes it before touching
 * the DSP boot regs. It is not a clock enable, despite the macro name.
 */
#define F286_AON_CLK_EN 0x006c
#define F286_AON_LTE_BOOT_ADDR 0x0248
#define F286_AON_GSM_BOOT_ADDR 0x0250
#define F286_AON_LTE_START 0x024c
#define F286_AON_GSM_START 0x0254
/* AON APB 0x402E block: one register space; +0x1000/+0x2000 are the
 * per-register SET/CLR alias banks of the same registers. The AON2/AON3
 * names below are historical: they are not separate banks.
 */
#define F286_AON_GATE2_04 \
	0x1004 /* APB_EB1 SET (b8 CA53_WDG_EB; 16-bit stores) */
#define F286_AON_GATE2_10 0x1010 /* APB_RTC_EB SET (b9 CA53_WDG_RTC_EB) */
#define F286_AON_GATE3_04 0x2004 /* APB_EB1 CLR */
#define F286_AON_GATE3_10 0x2010 /* APB_RTC_EB CLR */
#define F286_AON3_CLR_144 0x2144 /* AUTO_GATE_CTRL1 CLR alias (0x402e2144) */
/* AUTO_GATE_CTRL1 SET alias 0x402e1144 (reset value 0x15): writing 0x1a
 * SETs b1 cgm_wcdma_wtlcp_force_en + b3 cgm_wcdma_pubcp_force_en + b4
 * cgm_wcdma_root_auto_gate_sel = force-enable of the WCDMA pubcp+wtlcp
 * clocks. It is not a DSP interrupt enable; the AP-side DSP interrupt lines
 * are IRQs 113/115.
 */
#define F286_AON_WCDMA_FORCE 0x1144
/* APB_EB2 aliases (reset value 0x8071): SET b0 = WCDMA_EB (clock enable).
 * The 0x20b0 = 0xf tail CLEARS b3:0 = RFTI_RX_EB/RFTI_TX_EB/WCDMA_ICI_EB/
 * WCDMA_EB, i.e. it gates the WCDMA APB clock off. That contradicts the
 * "bridge" macro names, but the sequence with this tail has powered the die
 * on hardware, so it stays as it is.
 */
#define F286_AON2_BRIDGE 0x10b0 /* APB_EB2 SET b0 (WCDMA_EB) */
#define F286_AON3_BRIDGE 0x20b0 /* APB_EB2 CLR b3:0 (stock tail; see above) */
#define F286_AON_LTE_0244 0x0244

/* Keyed 0x40310000 block - 16-bit access ONLY. */
#define F286_KEY_EN 0x08
#define F286_KEY_KEY 0x20
#define F286_KEY_OPEN 0xe551
#define F286_KEY_CLOSE 0x1aae

/* DSP AP-window command area (window-relative). */
#define F286_Q_CTL 0x0e400 /* u16 in (low), u16 out (high) */
#define F286_Q_ENTRY 0x0e404 /* 8 entries x 8 bytes */
#define F286_Q_BODY 0x0e4e0 /* body area start, 0xc80 bytes */
#define F286_Q_BODY_SPAN 0x00c80
#define F286_Q_WATCH 0x00c00 /* bytes watched by the poller */
#define F286_Q_USER_SPAN 0x01c00 /* per window via the queue bin_attr */

/* Resp queue (from the stock reader at 0x80c10952): a separate ring
 * behind the 8 cmd entries; ctl in @0xE478 / out @0xE47A,
 * 12 x 8 B entries @0xE47C = { u16 resp_type; u16 resp_len; u32 resp_addr },
 * payload at win + 0xFBEC + resp_addr (sanity: addr + len < 0x100C).
 */
#define F286_Q_RSP_CTL 0x0e478 /* u16 in (low), u16 out (high) */
#define F286_Q_RSP_ENTRY 0x0e47c /* 12 entries x 8 bytes */
#define F286_Q_RSP_BODY 0x0fbec /* resp payload base (window-rel) */
#define F286_Q_RSP_MOD 12 /* ring modulus */

/* Split-path geometry (stock 0x81181d8c / 0x811816f0, model/l1ring.h):
 * payloads > 0x190 ride in 4 x 0x2800 slots at LTE window +0x33b000 (the
 * 10 x 0x2800 UL-data slots at +0x345000 are mapped too, span to +0x360000);
 * the ring body carries only the 8-byte header (marker 0xaaaa) + u16 slot
 * handle at body+8; entry.len stays the LOGICAL payload+8.
 */
#define F286_Q_INLINE_MAX 0x190 /* inline payload cap (stock INLINE_MAX) */
#define F286_LTE_SPLIT_OFF 0x33b000 /* SPLIT_BODY_OFF, abs 0x8073b000 */
#define F286_LTE_SPLIT_SPAN 0x025000 /* ..+0x360000: split + UL-data slots */
#define F286_LTE_SPLIT_SLOT 0x2800
#define F286_LTE_SPLIT_SLOTS 4

/* Stock's first LTE command: cmdId 6, 8-byte template from kernel RAM
 * 0x81717514 (init-data image offset 0x17514), verified: all zero bytes.
 */
static const u8 f286_lte_cmd6_template[8] = {};

#define F286_GSM_FW_BYTES 0x00300000UL
#define F286_LTE_FW_BYTES 0x002e4000UL

enum f286_state {
	F286_ST_OFF,
	F286_ST_POWERING,
	F286_ST_POWER_FAILED,
	F286_ST_POWER_OK,
	F286_ST_GSM_RUNNING,
	F286_ST_LTE_RUNNING,
};

static const char *const f286_state_names[] = {
	"off",	    "powering",	   "power-failed",
	"power-ok", "gsm-running", "lte-running",
};

/* boot_addr_mode (read at every load, runtime-writable):
 * 0 = the stock values, entry >> 1 (the HAL writes its argument verbatim
 *     and the caller shifts; LTE 0x40224200 = 0x80448400>>1, GSM
 *     0x40080040 = 0x80100080>>1), the default;
 * 1 = raw entry (LTE 0x80448400, GSM 0x80100080);
 * 2 = corrected + 4 (LTE 0x40224204 = 0x80448408>>1, GSM 0x40080044).
 */
static unsigned int boot_addr_mode;
module_param(boot_addr_mode, uint, 0644);
MODULE_PARM_DESC(boot_addr_mode,
		 "0 = corrected >>1 (stock), 1 = raw entry, 2 = corrected+4");

static u32 f286_lte_boot_addr(void)
{
	unsigned int mode = boot_addr_mode;

	switch (mode) {
	case 1:
		return 0x80448400u;
	case 2:
		return 0x40224204u;
	default:
		return 0x40224200u;
	}
}

/* GSM entry = image + 0x80 = 0x80100080. Stock 0x810b343c computes
 * ([0x81712f1c] + 0x20080) >> 1 with [0x81712f1c] = 0x800e0000 (window
 * base, not the image base), and the emulated stock boot writes 0x40080040.
 * An older value, 0x40090040, came from assuming the base held 0x80100000.
 */
static u32 f286_gsm_boot_addr(void)
{
	unsigned int mode = boot_addr_mode;

	switch (mode) {
	case 1:
		return 0x80100080u;
	case 2:
		return 0x40080044u;
	default:
		return 0x40080040u;
	}
}

/* stage=1: load_lte/load_gsm stop right after mapping their AP window. */
static bool stage;
module_param(stage, bool, 0444);
MODULE_PARM_DESC(stage,
		 "1 = load_lte/load_gsm stop after mapping the AP window");

/* dieclk=1: opt-in guarded die-local clock release (0xD15D/E). Default
 * off: the 0xD15E0028 pre-probe read hung the bus on hardware even with the
 * die powered; the whale AON block is not AP-reachable in our state, reason
 * unknown. Runtime-writable: echo 1 before load_lte.
 */
static bool dieclk;
module_param(dieclk, bool, 0644);
MODULE_PARM_DESC(
	dieclk,
	"1 = attempt the guarded 0xD15D/E die-local clock release during load (default off: a 0xD15E0028 read hung the bus on hardware)");

/* e010_light=1: opt-in 0xE0100000 light-path access. Default off: the
 * 0xE010 die-local family hung the bus on hardware (isolated single-access
 * probes), the same hang class as dieclk's 0xD15E. The read+RMW+write trio
 * in xdsp_clock_on sits on the load_lte path under rad->lock, so a dead-bus
 * read there wedged the whole sequence. Runtime-writable: echo 1 before
 * load_lte; when trying it, do one probe access per boot.
 */
static bool e010_light;
module_param(e010_light, bool, 0644);
MODULE_PARM_DESC(
	e010_light,
	"1 = touch the 0xE010100C/1014 light-path in xdsp_clock_on (default off: the 0xE010 family hung the bus on hardware)");

/* unlock=1 (default): run the unlock package in radio_power right after
 * keyed step-0: stock opener 0x8136F558 SETS [0x40310008] bit2 (an ORR; our
 * step-0 only clears {0,1,3}) inside its own
 * keyed window (0xE551 -> [0x40310020], write, close 0xFFFF1AAE), then the
 * AON writes that are real instructions in live functions: [0x402E1020]=
 * 0x40000 / [0x402E2020]=0xC0000 (selector 0x80A072E4, reached through the
 * call chain 0x80A091C6->0x80A06FF8) and [0x402E1134]=0x1000
 * (0x81252240 bring-up twin).
 */
static bool unlock = true;
module_param(unlock, bool, 0644);
MODULE_PARM_DESC(
	unlock,
	"1 = run the unlock package in radio_power (opener [0x40310008]|=4 + 0x402E1020/2020/1134)");

/* send_cmd6 (default 0): the stock cold boot sends no command to the DSP;
 * the handshake is DSP-initiated (unprompted resp type 5
 * version, drained by the pump kthread). The post-release cmdId-6 send is kept
 * only for experiments.
 */
static bool send_cmd6;
module_param(send_cmd6, bool, 0644);
MODULE_PARM_DESC(
	send_cmd6,
	"1 = send the post-release cmdId-6 template (stock cold boot sends nothing; resp type 5 is the handshake)");

/* hdr_layout: body-header shape for headered sends. Stock builder
 * 0x80e03c9a writes the 8-byte header as
 * [body+0]=desc[6] u16 (strh @0x80e03cf4), [body+2]=0x5555 (@0x80e03dbc),
 * [body+4]=desc[0] u32 (str.w @0x80e03cec); desc[0] is threaded from the
 * send entry 0x80e0412e (desc fill @0x80e040c8-ce) and desc[6] carries
 * the command id (the same id drives the wire-length lookup 0x80e038a0).
 * Whether the DSP decodes cmdId at +0 or +4 stays ambiguous, so:
 * 0 = legacy driver form {u16 id, 0x5555, u32 0};
 * 1 = stock-desc (default, the most likely reading): the command id goes into
 *     BOTH desc slots, {u16 id, 0x5555, u32 id}, so the DSP finds cmdId at
 *     +0 or +4 under either reading.
 * Fixed-8 sends (cmd 6/0xd) carry no message header at all (raw8).
 */
static unsigned int hdr_layout = 1;
module_param(hdr_layout, uint, 0644);
MODULE_PARM_DESC(
	hdr_layout,
	"body header: 0 = legacy {id,5555,0}, 1 = stock-desc {id,5555,id}");

/* xdsp_mux (model/clocks.h xdsp_clk_mux_select): XDSP exec-clock source
 * mux at AP clock mgr 0x402D02B4; sel 2 = TWPLL_51M2 (51.2 MHz), the DSP
 * exec clock. Read, written only when the current sel differs, logged with
 * readback.
 */
static unsigned int xdsp_mux = 2;
module_param(xdsp_mux, uint, 0644);
MODULE_PARM_DESC(xdsp_mux,
		 "0x402D02B4 XDSP clock mux selector (2 = TWPLL_51M2)");

/* stock_early=1: opt-in remaining stock first-second writes missing from
 * radio_power (emulated stock boot at 0.89-0.9 s, docs/bringup-status.md): the
 * PMU power-domain pairs (SET b25 / CLEAR b24) on 0x28/0x2c/0x3c/0x4c/0x50,
 * 0x402b0210 0->1, and [0x402e0244]=0x10100 / [0x404d0008]=0x8c (removed
 * earlier as "dead code", but the stock boot executes both). DEFAULT OFF.
 */
static bool stock_early;
module_param(stock_early, bool, 0644);
MODULE_PARM_DESC(
	stock_early,
	"1 = add the remaining stock first-second writes to radio_power (PMU domain pairs, 0x402b0210, 0x402e0244, 0x404d0008)");

/* xdsp_cores=1: load_lte boots the two XDSP cores (windows 0xE0400000 /
 * 0xE0480000) from f286-xdsp{A,B}-{boot,main}.bin the way stock does at
 * ~0.97 s, after the clock-on and before the LTE boot-addr. The images exist
 * only inside the stock kernel.bin (tools/extract_dsp_fw.py cuts them out).
 * Default off: the 0xE0xx die range has hung the bus before, and this is
 * the driver's only access to 0xE04xxxxx.
 */
static bool xdsp_cores;
module_param(xdsp_cores, bool, 0644);
MODULE_PARM_DESC(
	xdsp_cores,
	"1 = load_lte boots XDSP cores A/B (stock boot loader -> main, ready 1 -> 2) before the LTE release");

/* gsm_iram=1: load_gsm also does the stock writes to low on-chip memory:
 * zero 0x10c00 (0xe00 bytes) + [0x10c00+0xdcc]=1 (L1 init 0x80e4c6e4) and
 * the mode cell 0x11d30 = 1 with read-back (0x80b18218). Whether Linux can
 * reach 0x10000-0x12000 is unknown, hence DEFAULT OFF.
 */
static bool gsm_iram;
module_param(gsm_iram, bool, 0644);
MODULE_PARM_DESC(
	gsm_iram,
	"1 = load_gsm writes the stock IRAM cells (0x10c00 clear, mode cell 0x11d30=1)");

/* autoint defaults OFF: the 60 s work only creates the remaining attrs and
 * (with autoint=1) runs the radio sequence. Normally the owner drives
 * everything through the attributes after SSH is up.
 */
static bool autoint;
static bool no_autoint;
module_param(autoint, bool, 0444);
module_param(no_autoint, bool, 0444);
MODULE_PARM_DESC(
	autoint,
	"deferred work also runs the radio power-on at +60 s (default off)");
MODULE_PARM_DESC(no_autoint,
		 "override: the deferred work only creates attributes");

struct f286_radio {
	struct device *dev;
	struct mutex lock;
	void __iomem *adi;
	void __iomem *pmu;
	void __iomem *aon;
	void __iomem *key;
	void __iomem *whale_apb;
	void __iomem *clkm; /* AP clock mgr 0x402D0000 (XDSP mux 0x402D02B4) */
	void __iomem *wcfg_e010; /* whale misc die cfg 0xE0100000 (identity) */
	void __iomem *waon_d; /* whale RF gate 0xD15D0000 (guarded only) */
	void __iomem *waon_e; /* whale AON/PMU 0xD15E0000 (guarded only) */
	void __iomem *blk404d;
	void __iomem *gicd;
	void __iomem *gsm_win;
	bool gsm_win_ioun;
	void __iomem *lte_win;
	bool lte_win_ioun;
	void __iomem *hs_win; /* LTE handshake window 0x80447000 */
	bool hs_win_ioun;
	void __iomem *split_win; /* LTE split slots, 0x8073b000 */
	bool split_win_ioun;
	const char *fw_names[2]; /* [0] = LTE, [1] = GSM */
	enum f286_state state;
	struct task_struct *loader; /* in-flight load+poll thread */
	struct task_struct *powerer; /* in-flight radio_power thread */
	struct task_struct *
		dieclk_thread; /* non-NULL = dieclk kthread alive (uses its own private maps) */
	struct task_struct *pumper; /* persistent L1 resp-pump kthread */
	bool l1_type5; /* pump saw a resp type 5 (handshake) */
	u8 lte_split_idx; /* round-robin split-slot index (mod 4) */
	bool load_lte_req; /* loader target: true = LTE, false = GSM */
	bool gicd_masked; /* originals below are the pre-mask values */
	u32 gicd_icenabler2,
		gicd_icenabler3; /* pre-mask GICD_ICENABLER2/3 for the remove restore */
	bool poll_seen_change;
	bool init_done;
	bool sysfs_ready;
	struct delayed_work init_work;
};

static int f286_init_locked(struct f286_radio *rad);

static void __iomem *f286_map_mmio(struct device *dev, unsigned long phys,
				   size_t span)
{
	void __iomem *p = ioremap(phys, span);

	if (!p)
		dev_err(dev, "ioremap %08lx+%zx failed\n", phys, span);
	return p;
}

static void f286_unmap_mmio(void __iomem **p)
{
	if (*p) {
		iounmap(*p);
		*p = NULL;
	}
}

/*
 * DSP DRAM windows (all below the Linux memory base 0x80760000 - the DT
 * memory node sits above the whole DSP region including the LTE UL-data
 * slots that reach 0x8075E000). memremap(MEMREMAP_WC) is
 * preferred; ioremap_wc is the fallback. WC keeps the stores uncached so
 * the non-cache-coherent DSP cores observe firmware/queue writes. The
 * printk pair brackets the mapping: if it ever kills the kernel, the panic
 * point is the last console line.
 */
static void __iomem *f286_map_dram(struct device *dev, unsigned long phys,
				   size_t span, bool *ioun)
{
	phys_addr_t pa = phys;
	void __iomem *p;

	*ioun = false;
	dev_info(dev,
		 "map dram %pap size %zx (memremap WC, ioremap_wc fallback)\n",
		 &pa, span);
	p = (__force void __iomem *)memremap(phys, span, MEMREMAP_WC);
	if (p) {
		dev_info(dev, "map dram %pap memremap ok -> %p\n", &pa, p);
		return p;
	}
	p = ioremap_wc(phys, span);
	if (p) {
		*ioun = true;
		dev_info(dev, "map dram %pap ioremap_wc ok -> %p\n", &pa, p);
		return p;
	}
	dev_err(dev, "map dram %pap+%zx failed (memremap and ioremap_wc)\n",
		&pa, span);
	return NULL;
}

static void f286_unmap_dram(void __iomem **p, bool ioun)
{
	if (!*p)
		return;
	if (ioun)
		iounmap(*p);
	else
		memunmap((__force void *)*p);
	*p = NULL;
}

static int f286_get_mmio_locked(struct f286_radio *rad, void __iomem **slot,
				unsigned long phys, size_t span,
				const char *name)
{
	if (*slot)
		return 0;
	*slot = f286_map_mmio(rad->dev, phys, span);
	if (!*slot) {
		dev_err(rad->dev, "%s (%08lx) unmappable, op unavailable\n",
			name, phys);
		return -EIO;
	}
	return 0;
}

static int f286_get_dram_locked(struct f286_radio *rad, void __iomem **slot,
				bool *ioun, unsigned long phys, size_t span,
				const char *name)
{
	if (*slot)
		return 0;
	*slot = f286_map_dram(rad->dev, phys, span, ioun);
	if (!*slot) {
		dev_err(rad->dev, "%s (%08lx) unmappable, op unavailable\n",
			name, phys);
		return -EIO;
	}
	return 0;
}

/* RMW with a physical label for the log. */
static void f286_rmw32(struct f286_radio *rad, const char *label,
		       void __iomem *base, unsigned int off, u32 set, u32 clear)
{
	u32 old = readl(base + off);
	u32 new = (old | set) & ~clear;

	writel(new, base + off);
	dev_info(rad->dev, "RMW %s[%03x]: %08x -> %08x (set %08x clr %08x)\n",
		 label, off, old, new, set, clear);
}

/* ---- ADI (analog die) access; protocol as in spi-sprd-adi.c ---- */

static int f286_adi_lock(struct f286_radio *rad)
{
	u32 val;
	int ret;

	ret = readl_poll_timeout(rad->adi + F286_ADI_USER_LOCK, val, val == 0,
				 1000, 50000);
	if (ret) {
		dev_warn(rad->dev,
			 "ADI user-lock 0x224 busy (%08x), access skipped\n",
			 val);
		return -EBUSY;
	}
	return 0;
}

static void f286_adi_unlock(struct f286_radio *rad)
{
	writel(F286_ADI_UNLOCK_TOKEN, rad->adi + F286_ADI_USER_LOCK);
}

static int f286_adi_read16(struct f286_radio *rad, u32 reg, u16 *out)
{
	u32 val;
	int ret;

	writel(reg & 0xffffu, rad->adi + F286_ADI_RD_CMD);
	ret = readl_poll_timeout(rad->adi + F286_ADI_RD_DATA, val,
				 !(val & F286_ADI_RD_BUSY), 10, 20000);
	if (ret)
		return ret;
	*out = val & 0xffffu;
	return 0;
}

static int f286_adi_write16(struct f286_radio *rad, u32 reg, u16 value)
{
	u32 val;
	int ret;

	ret = f286_adi_lock(rad);
	if (ret)
		return ret;
	readl_poll_timeout(rad->adi + F286_ADI_ARM_FIFO_STS, val,
			   val & F286_ADI_FIFO_EMPTY, 10, 50000);
	ret = readl_poll_timeout(rad->adi + F286_ADI_ARM_FIFO_STS, val,
				 !(val & F286_ADI_FIFO_FULL), 10, 20000);
	if (ret == 0) {
		writel(value, rad->adi + (reg - F286_ADI_PHYS));
		readl_poll_timeout(rad->adi + F286_ADI_ARM_FIFO_STS, val,
				   val & F286_ADI_FIFO_EMPTY, 10, 50000);
	}
	f286_adi_unlock(rad);
	return ret;
}

static int f286_adi_rmw16(struct f286_radio *rad, u32 reg, u32 set, u32 clear)
{
	u16 old = 0, rb = 0;
	u16 new;
	int ret;

	ret = f286_adi_read16(rad, reg, &old);
	if (ret) {
		dev_err(rad->dev, "ADI RMW %08x read failed (%pe)\n", reg,
			ERR_PTR(ret));
		return ret;
	}
	new = ((old | set) & ~clear) & 0xffffu;
	ret = f286_adi_write16(rad, reg, new);
	if (ret) {
		dev_err(rad->dev, "ADI RMW %08x write failed (%pe)\n", reg,
			ERR_PTR(ret));
		return ret;
	}
	f286_adi_read16(rad, reg, &rb);
	dev_info(rad->dev, "ADI RMW %08x: %04x -> %04x (rb %04x)\n", reg, old,
		 new, rb);
	return 0;
}

/* The stock analog bring-up batch (0x811d05b4) + radio-on RF tail. */
static const struct {
	u32 reg;
	u32 set;
	u32 clear;
} f286_analog_batch[] = {
	{ 0x40608c08, 0x20, 0 },  { 0x40608c0c, 0x40, 0 },
	{ 0x40608c0c, 0x20, 0 },  { 0x40608de8, 0x100, 0 },
	{ 0x40608454, 0x1, 0 },	  { 0x40608458, 0x1, 0 },
	{ 0x40608404, 0x1, 0 },	  { 0x40608404, 0x1000, 0 },
	{ 0x40608404, 0, 0x700 }, { 0x40608e88, 0x20, 0x8 },
};

/* Keyed step-0, 16-bit stores; caller verified the gates latched (an
 * oopsing store leaks the kernfs active ref and locks the subtree).
 */
static void f286_key_step0(struct f286_radio *rad)
{
	u16 before, after;

	writew(F286_KEY_OPEN, rad->key + F286_KEY_KEY);
	before = readw(rad->key + F286_KEY_EN);
	writew(before & 0xfff4u, rad->key + F286_KEY_EN);
	after = readw(rad->key + F286_KEY_EN);
	writew(F286_KEY_CLOSE, rad->key + F286_KEY_KEY);
	dev_info(rad->dev, "keyed step-0: 0x40310008 %04x -> %04x (16-bit)\n",
		 before, after);
}

/* ---- whale-die power-on (stock 0x80d0f186 order; works on hardware) ---- */

/* Stock re-poll fn 0x80d0eb0c: d4[7:4]
 * triple-read-stable ==6, budget 0x2710 samples (movw sl,#0x2710
 * @0x80d0eb18). Its success tail is whale 0x60830044 |= 0x80 + 0x50 +
 * die-local [0x5150D000+0x18] read + 0x50. The |=0x80 runs
 * unconditionally as step E right after the wait below; the 0x5150D018
 * read is deliberately not replicated (an unguarded die-local touch is the
 * bus-hang class described at dieclk). Only the verdict is returned.
 */
static bool f286_wtlcp_repoll(struct f286_radio *rad, ktime_t deadline)
{
	unsigned int i;

	for (i = 0; i < 0x2710; i++) {
		u32 da = readl(rad->pmu + F286_PMU_WHALE_FSM);
		u32 db = readl(rad->pmu + F286_PMU_WHALE_FSM);
		u32 dc = readl(rad->pmu + F286_PMU_WHALE_FSM);

		if (((da >> 4) & 0xf) == 6 && ((db >> 4) & 0xf) == 6 &&
		    ((dc >> 4) & 0xf) == 6)
			return true;
		if (kthread_should_stop())
			return false; /* removable mid-wait */
		if (ktime_after(ktime_get(), deadline))
			break;
		/* usleep_range is TASK_UNINTERRUPTIBLE; use the
		 * stoppable sleep. Success path only (the triple read is
		 * already stable when this fn is entered), so the short
		 * lock-held window left here is bounded and rare.
		 */
		msleep_interruptible(2);
	}
	return false;
}

/* Stock settle (checked by hand against the assembly): every whale
 * 0x60830044 enable/RMW helper is followed by bus.spin(0x50) - the veneer
 * 0x80D0E90C calls write-helper 0x811750b2 and the 0x50-iteration delay loop
 * sits at 0x80d0e912 (cmp r0,#0x50 / blo). Stock never issues back-to-back
 * whale-APB accesses without it. Driver approximate: 0x50-iteration barrier
 * loop before the next register access.
 */
static void f286_whale_settle(struct f286_radio *rad)
{
	unsigned int i;

	for (i = 0; i < 0x50; i++)
		barrier();
	dev_info(rad->dev, "whale settle: spin 0x50 done\n");
}

static int f286_radio_power_locked(struct f286_radio *rad)
{
	u32 c0, c4, d4, before;
	unsigned int i, round;
	int ok = 0, latched;
	ktime_t deadline;
	int ret;

	ret = f286_get_mmio_locked(rad, &rad->pmu, F286_PMU_PHYS, F286_PMU_SPAN,
				   "pmu-0x402b");
	if (!ret)
		ret = f286_get_mmio_locked(rad, &rad->aon, F286_AON_PHYS,
					   F286_AON_SPAN, "aon-0x402e");
	if (!ret)
		ret = f286_get_mmio_locked(rad, &rad->adi, F286_ADI_PHYS,
					   F286_ADI_SPAN, "adi-0x4060");
	if (!ret)
		ret = f286_get_mmio_locked(rad, &rad->key, F286_KEY_PHYS,
					   F286_KEY_SPAN, "key-0x4031");
	if (!ret)
		ret = f286_get_mmio_locked(rad, &rad->whale_apb,
					   F286_WHALE_APB_PHYS,
					   F286_WHALE_APB_SPAN, "whale-0x6083");
	if (!ret)
		ret = f286_get_mmio_locked(rad, &rad->blk404d,
					   F286_BLK404D_PHYS, F286_BLK404D_SPAN,
					   "blk-0x404d");
	if (ret)
		return ret;

	/* GICD IRQ mask before the sequence (best-effort). 0xe0000030 =
	 * SPIs 37,61-63 -> hwirq 68/69/93/94/95, but my reading of the stock
	 * SIPC/WCN code puts hwirq 68/69 on the WCN mailbox (sprd-mailbox), so
	 * the SPI numbering is in doubt. The mask is left as it is for now; it
	 * must be settled before any BT/WCN work.
	 */
	if (f286_get_mmio_locked(rad, &rad->gicd, F286_GICD_PHYS,
				 F286_GICD_SPAN, "gicd") == 0) {
		/* Keep the original ICENABLER2/3 values: f286_radio_remove
		 * restores them before the unmap so the mask is reversible at
		 * driver release (the masked set plausibly includes hwirq
		 * 68/69 = the WCN/BT mailbox). Save only the FIRST mask of
		 * this boot: a later re-mask would otherwise snapshot the
		 * already-masked value.
		 */
		before = readl(rad->gicd + 0x188);
		if (!rad->gicd_masked) {
			rad->gicd_icenabler2 = before;
			rad->gicd_icenabler3 = readl(rad->gicd + 0x18c);
			rad->gicd_masked = true;
		}
		writel(0xe0000030, rad->gicd + 0x188);
		writel(0xe0000000, rad->gicd + 0x18c);
		dev_info(
			rad->dev,
			"masked unclaimed baseband IRQs: GICD_ICENABLER2 %08x->%08x, ICENABLER3 %08x->e0000000 (originals saved for remove restore)\n",
			before, readl(rad->gicd + 0x188), rad->gicd_icenabler3);
	} else {
		dev_warn(
			rad->dev,
			"GICD unmappable: baseband IRQ mask SKIPPED (unclaimed DSP IRQs may storm)\n");
	}

	rad->state = F286_ST_POWERING;

	dev_info(
		rad->dev,
		"power: A. AON gate pairs (16-bit SET stores + latch verify)\n");
	writew(0x100, rad->aon + F286_AON_GATE2_04);
	writew(0x200, rad->aon + F286_AON_GATE2_10);
	writew(0x100, rad->aon + F286_AON_GATE3_04);
	writew(0x200, rad->aon + F286_AON_GATE3_10);

	/* Verify the gates LATCHED before any keyed access (a 32-bit SET
	 * store is silently dropped and an ungated keyed store bus-aborts).
	 */
	latched = (readl(rad->aon + 0x004) & 0x100) &&
		  (readl(rad->aon + 0x010) & 0x200);
	dev_info(
		rad->dev,
		"power: gate latch 0x402e0004=%08x (bit8=%u) 0x402e0010=%08x (bit9=%u) aon3 %08x/%08x\n",
		readl(rad->aon + 0x004), (readl(rad->aon + 0x004) >> 8) & 1,
		readl(rad->aon + 0x010), (readl(rad->aon + 0x010) >> 9) & 1,
		readl(rad->aon + F286_AON_GATE3_04),
		readl(rad->aon + F286_AON_GATE3_10));
	if (!latched) {
		u16 v;

		dev_warn(
			rad->dev,
			"power: SET aliases did not latch; retrying via 16-bit RMW on the BASE regs\n");
		v = readw(rad->aon + 0x004);
		writew(v | 0x100, rad->aon + 0x004);
		v = readw(rad->aon + 0x010);
		writew(v | 0x200, rad->aon + 0x010);
		latched = (readl(rad->aon + 0x004) & 0x100) &&
			  (readl(rad->aon + 0x010) & 0x200);
		dev_info(rad->dev, "power: base-RMW latch retry -> %d\n",
			 latched);
	}
	if (latched) {
		dev_info(rad->dev, "power: B. keyed step-0 (gates verified)\n");
		f286_key_step0(rad);
		if (unlock) {
			/* Unlock package: stock opener 0x8136F558 SETS
			 * [0x40310008] bit2 (orr #4) inside its keyed
			 * window; keyed step-0 above only
			 * CLEARS {0,1,3} and never sets bit2. Then the AON
			 * writes that are real instructions in live code:
			 * [0x402E1020]=0x40000 / [0x402E2020]=0xC0000
			 * (selector 0x80A072E4, reached through
			 * 0x80A091C6->0x80A06FF8) and
			 * [0x402E1134]=0x1000 (0x81252240 bring-up twin). The
			 * opener's own gate pair ([0x402E1004]=0x100 /
			 * [0x402E1010]=0x200) is the block-A gate set above.
			 * Breadcrumb before each write.
			 */
			u16 kbefore, kafter;

			dev_info(rad->dev, "power: unlock package\n");
			writew(F286_KEY_OPEN, rad->key + F286_KEY_KEY);
			kbefore = readw(rad->key + F286_KEY_EN);
			writew(kbefore | 4u, rad->key + F286_KEY_EN);
			kafter = readw(rad->key + F286_KEY_EN);
			writew(F286_KEY_CLOSE, rad->key + F286_KEY_KEY);
			dev_info(
				rad->dev,
				"power: opener 0x8136F558 [0x40310008] |= 4 (keyed 16-bit): %04x -> %04x, key closed\n",
				kbefore, kafter);
			writel(0x40000, rad->aon + 0x1020);
			dev_info(
				rad->dev,
				"power: unlock [0x402E1020] <- 40000 (0x80A072E4), rb %08x\n",
				readl(rad->aon + 0x1020));
			writel(0xc0000, rad->aon + 0x2020);
			dev_info(
				rad->dev,
				"power: unlock [0x402E2020] <- c0000 (0x80A072E4), rb %08x\n",
				readl(rad->aon + 0x2020));
			writel(0x1000, rad->aon + 0x1134);
			dev_info(
				rad->dev,
				"power: unlock [0x402E1134] <- 1000 (0x81252240), rb %08x\n",
				readl(rad->aon + 0x1134));
		} else {
			dev_info(rad->dev,
				 "power: unlock package SKIPPED (unlock=0)\n");
		}
	} else {
		dev_warn(
			rad->dev,
			"power: gates did not latch, skipping keyed step-0 (necessity unproven; continuing without it)\n");
	}

	dev_info(rad->dev, "power: C. analog ADI batch (%zu ops)\n",
		 ARRAY_SIZE(f286_analog_batch));
	for (i = 0; i < ARRAY_SIZE(f286_analog_batch); i++)
		f286_adi_rmw16(rad, f286_analog_batch[i].reg,
			       f286_analog_batch[i].set,
			       f286_analog_batch[i].clear);

	dev_info(rad->dev, "power: D. radio-on set (stock 0x80d0f186 order)\n");
	/* The first sequence that actually powered the die on hardware
	 * cleared the DSP shared deep-sleep force bit17 first (a fresh boot
	 * has it set: baseline 0x402b00cc = 0x002200C2), then bit21.
	 * Skipping the bit17 clear leaves the released DSP held in forced
	 * deep sleep. Bank2 CLEAR alias, same style as the bit21 write.
	 */
	writel(0x00020000, rad->pmu + F286_PMU_CLR_DSP_CFG); /* DSP bit17 */
	writel(0x00200000, rad->pmu + F286_PMU_CLR_DSP_CFG); /* CM4 bit21 */
	/* CP_SOFT_RST release (same write as stock): 0x402b00b0 resets to
	 * 0x507 with WTLCP_SOFT_RST b0 + WTLCP_DSP_SYS_SRST b2 held; stock GSM
	 * pre-boot 0x80D0FA5E and the OS sweep 0x80D0FAAA both write 5 to the
	 * CLR bank. Without this the driver silently depended on an earlier
	 * boot stage having done it.
	 */
	writel(0x00000005, rad->pmu + F286_PMU_CLR_CPSOFT_RST);
	dev_info(
		rad->dev,
		"power: 0x402b20cc <- 020000+200000 (bit17 first), rb 0x402b00cc=%08x\n",
		readl(rad->pmu + F286_PMU_RD_00CC));
	writel(0x10, rad->aon + F286_AON3_CLR_144); /* AUTO_GATE_CTRL1 CLR b4 */
	writel(0x1, rad->aon + F286_AON3_CLR_144); /* AUTO_GATE_CTRL1 CLR b0 */
	writel(0x2,
	       rad->aon + F286_AON_WCDMA_FORCE); /* AUTO_GATE_CTRL1 SET b1 */
	writel(0x1, rad->aon + F286_AON2_BRIDGE); /* APB_EB2 SET WCDMA_EB */
	writel(0xf,
	       rad->aon + F286_AON3_BRIDGE); /* stock tail; see define NOTE */
	/* [0x402e0244] = 0x00010100 and [0x404d0008] |= 0x80 were once
	 * dropped as dead code in the 0x80D0F186 chain, but the emulated stock
	 * boot executes both at 0.899 s; they are back under stock_early,
	 * together with the PMU domain pairs.
	 */
	if (stock_early) {
		/* Stock 0.898 s: each power-domain cfg gets SET b25 (force
		 * shutdown) then CLEAR b24 (auto shutdown) via the bank1/bank2
		 * aliases. 0x34 (LDSP) is released again by load_lte [2], 0x3c
		 * (WHu3g) by step F below, as stock does later.
		 */
		static const u16 pd_cfg[] = {
			0x28, 0x2c, 0x34, 0x3c, 0x4c, 0x50
		};

		for (i = 0; i < ARRAY_SIZE(pd_cfg); i++) {
			writel(0x02000000, rad->pmu + 0x1000 + pd_cfg[i]);
			writel(0x01000000, rad->pmu + 0x2000 + pd_cfg[i]);
			dev_info(
				rad->dev,
				"power: stock_early PD cfg 0x402b%04x set b25 / clr b24, rb %08x\n",
				pd_cfg[i], readl(rad->pmu + pd_cfg[i]));
		}
	}
	writel(0x02000000, rad->pmu + F286_PMU_CLR_2044);
	writel(0x01000000, rad->pmu + F286_PMU_CLR_2044);
	f286_rmw32(rad, "0x60830044", rad->whale_apb, 0x044, 0x40, 0);
	if (stock_early) {
		writel(0x00010100, rad->aon + F286_AON_LTE_0244);
		writel(0, rad->pmu + 0x210);
		writel(1, rad->pmu + 0x210);
		f286_rmw32(rad, "0x404d0008", rad->blk404d, 0x008, 0x80, 0);
		dev_info(
			rad->dev,
			"power: stock_early 0x402e0244 <- 10100 rb %08x, 0x402b0210 <- 0,1 rb %08x\n",
			readl(rad->aon + F286_AON_LTE_0244),
			readl(rad->pmu + 0x210));
	}
	f286_whale_settle(
		rad); /* stock 0x80D0E90C/0x80D0E95C helpers spin 0x50 */

	/* WTLCP wait, shaped like stock: main poll
	 * (0x80d0ebd8) needs c4[4:0]==0 AND d4[7:4]==6 stable across THREE
	 * reads (0x80d0ec34-4a), budget 0x2710 samples; on success stock
	 * calls the re-poll fn 0x80d0eb0c (helper above); the timeout paths
	 * force-retry FOREVER (0x80d0ecd4 jumps back into the poll loop)
	 * with 0x402b2044=0x02000000 then =0x01000000 + whale 0x60830044 |=
	 * 0x40. Driver-only deviation: a 30 s hard cap so radio_power can
	 * never wedge the box - logged LOUDLY when it fires.
	 */
	dev_info(
		rad->dev,
		"power: polling c4[4:0]==0 && d4[7:4]==6 (triple-read-stable, 0x2710-sample rounds + stock force-retry, 30 s cap)\n");
	deadline = ktime_add_ms(ktime_get(), 30000);
	for (round = 0;; round++) {
		ok = 0;
		for (i = 0; i < 0x2710; i++) {
			u32 da, db;

			c4 = readl(rad->pmu + F286_PMU_WTLCP_STATUS);
			da = readl(rad->pmu + F286_PMU_WHALE_FSM);
			db = readl(rad->pmu + F286_PMU_WHALE_FSM);
			d4 = readl(rad->pmu + F286_PMU_WHALE_FSM);
			if ((c4 & 0x1f) == 0 && ((da >> 4) & 0xf) == 6 &&
			    ((db >> 4) & 0xf) == 6 && ((d4 >> 4) & 0xf) == 6) {
				ok = f286_wtlcp_repoll(rad, deadline);
				break;
			}
			if (ktime_after(ktime_get(), deadline))
				break;
			/* The ~30 s wait must not sit on rad->lock:
			 * release around the sleep (the guard state,
			 * POWERING + powerer/loader pointers, keeps other
			 * sequences out), and make it stoppable so remove
			 * can kill a wedged wait (usleep_range would be
			 * TASK_UNINTERRUPTIBLE).
			 */
			mutex_unlock(&rad->lock);
			msleep_interruptible(2);
			mutex_lock(&rad->lock);
			if (kthread_should_stop()) {
				/* Deliberately not unlocked here: this
				 * fn is called with rad->lock held and returns
				 * with it held; the caller (f286_power_thread)
				 * does the unlock.
				 */
				dev_warn(
					rad->dev,
					"power: stopped mid-WTLCP wait, sequence aborted\n");
				rad->state = F286_ST_POWER_FAILED;
				return -EINTR;
			}
		}
		if (ok)
			break;
		if (ktime_after(ktime_get(), deadline)) {
			dev_err(rad->dev,
				"power: WTLCP WAIT HARD CAP 30 s HIT - c4=%08x d4=%08x after %u force round(s); stock would retry forever (0x80d0ecd4), driver gives up for safety\n",
				c4, d4, round);
			break;
		}
		/* Stock force branch: clears both 0x402b0044 b25:24
		 * via the bank2 alias, whales 0x60830044 |= 0x40, jumps back
		 * into the poll.
		 */
		dev_warn(
			rad->dev,
			"power: WTLCP round %u failed (c4=%08x d4=%08x), stock force-retry\n",
			round, c4, d4);
		writel(0x02000000, rad->pmu + F286_PMU_CLR_2044);
		writel(0x01000000, rad->pmu + F286_PMU_CLR_2044);
		f286_rmw32(rad, "0x60830044 force", rad->whale_apb, 0x044, 0x40,
			   0);
		f286_whale_settle(rad);
	}

	c0 = readl(rad->pmu + F286_PMU_POWER_STATUS);
	dev_info(
		rad->dev,
		"power: %s c0=%08x (WHu3g[4:0]=%u CM4[19:15]=%u) c4=%08x (WTLCP[4:0]=%u) d4=%08x (state=%u)\n",
		ok ? "OK" : "TIMEOUT", c0, c0 & 0x1f, (c0 >> 15) & 0x1f, c4,
		c4 & 0x1f, d4, (d4 >> 4) & 0xf);
	rad->state = ok ? F286_ST_POWER_OK : F286_ST_POWER_FAILED;
	/* Step E: whale-APB XDSP clock writes from the stock
	 * post-power block (0x60831034 = 0x410 per-poweron cfg, then |=0x80;
	 * 0x60831044 = 0x80 module enable). The 0x60830044 |= 0x80 here also
	 * covers the re-poll fn 0x80d0eb0c success tail; its die-local
	 * 0x5150D018 read stays skipped (see f286_wtlcp_repoll). On hardware
	 * the readback was 0 before the die was powered (sub-block bus clocked
	 * off). AP-reachable page; if rb is still 0, log and continue.
	 */
	dev_info(rad->dev, "power: E. whale XDSP clock writes (rb-check)\n");
	/* Stock re-poll fn 0x80d0eb0c success tail = the enable veneer
	 * 0x80D0E90C: [0x60830044] |= 0x80 + spin(0x50). The short offset is
	 * correct here (0x60830044 is the up-request register; the 0x60831xxx
	 * clock/int regs are the sub-block at +0x1000). Its die-local
	 * 0x5150D018 read stays skipped (see f286_wtlcp_repoll).
	 */
	f286_rmw32(rad, "0x60830044", rad->whale_apb, 0x044, 0x80, 0);
	f286_whale_settle(rad);
	/* The XDSP clock/int registers live in the 0x60831000 sub-block
	 * (stock code loads 0x60831000 as a base; model/clocks.cpp,
	 * model/pmu.cpp). The 0x60831xxx writes use +0x1000 offsets (the
	 * map span 0x3000 covers them). There is no "0x60831034 |= 0x80":
	 * the only stock +0x34 stores are =0x410 / =0x10 / =0x40000.
	 */
	writel(0x410, rad->whale_apb + 0x1034);
	dev_info(rad->dev, "power: 0x60831034 <- 0x410, rb %08x\n",
		 readl(rad->whale_apb + 0x1034));
	writel(0x80, rad->whale_apb + 0x1044);
	dev_info(rad->dev, "power: 0x60831044 <- 0x80, rb %08x\n",
		 readl(rad->whale_apb + 0x1044));
	dev_info(
		rad->dev,
		"power: die-local pages are at 0xD15D/E0000 (1:1 in stock) - handled by the dieclk kthread during load, not here\n");
	/* WHu3g island arm (stock whale_power_on 0x80D0F7A0..A6 +
	 * wait_whu3g_on 0x80D0ED6A, model/whale.cpp): clear
	 * PD_WTLCP_HU3GE_A_CFG b25 FORCE_SHUTDOWN via the bank2 CLR alias,
	 * then poll STATUS0[4:0] triple-read-stable == 0. Without it this
	 * only works by luck: the reset value 0x01208804 has b25=0, but stock
	 * writes the
	 * clear explicitly on every power-on (and b24 auto-shutdown stays
	 * armed until the field is touched). The stock RF-gate step
	 * [0xD15D0020] 0->2 rides in the guarded dieclk kthread only (it is
	 * in the bus-hang class). Stock polls up to 1e6 samples; driver caps at
	 * ~3 s and continues NON-fatally (stock's result is caller-logged).
	 */
	dev_info(rad->dev, "power: F. WHu3g arm (whale_power_on tail)\n");
	writel(0x02000000, rad->pmu + F286_PMU_CLR_203C); /* clr 003c b25 */
	{
		bool on = false;
		unsigned int tries;

		for (tries = 0; tries < 200; tries++) {
			u32 wa = readl(rad->pmu + F286_PMU_POWER_STATUS);
			u32 wb = readl(rad->pmu + F286_PMU_POWER_STATUS);
			u32 wc = readl(rad->pmu + F286_PMU_POWER_STATUS);

			if (!(wa & 0x1f) && !(wb & 0x1f) && !(wc & 0x1f)) {
				on = true;
				break;
			}
			/* Sleep with the lock released. */
			mutex_unlock(&rad->lock);
			msleep_interruptible(1);
			mutex_lock(&rad->lock);
			if (kthread_should_stop()) {
				dev_warn(
					rad->dev,
					"power: stopped mid-WHu3g wait, sequence aborted\n");
				rad->state = F286_ST_POWER_FAILED;
				return -EINTR;
			}
		}
		dev_info(rad->dev,
			 "power: whu3g c0[4:0]%s (%u samples, c0=%08x)\n",
			 on ? " == 0 stable" : " NOT 0 after 3 s cap",
			 tries + 1, readl(rad->pmu + F286_PMU_POWER_STATUS));
	}
	return ok ? 0 : -ETIMEDOUT;
}

/* ---- DSP AP-window command queue (stock alloc 0x81181d3c, put 0x81181edc) ---- */

/* Two distinct body shapes, plus the split variant (they are easy to
 * conflate):
 *   raw8 (fixed-8 cmds 6/0xd, stock boot tail 0x80e2d986): body = payload
 *     bytes ONLY, entry.len = 8, entry.cmdId = cmd - NO MsgHeader;
 *   headered (pal-style, builder 0x80e03c9a): body = { u16 id; u16 marker
 *     0x5555; u32 id per hdr_layout; payload[] }, entry.len = payload_len+8,
 *     entry.cmdId = 0;
 *   split (model/l1ring.h send_message/split_alloc_put, stock 0x81181d8c +
 *     handle 0x811816f0): payload > 0x190 lives in a 0x2800 slot at LTE win
 *     +0x33b000; ring body = header with marker 0xaaaa + u16 slot handle at
 *     body+8 (u16 pad), entry.len = LOGICAL payload_len+8.
 * The DSP polls the ring; no per-message doorbell exists.
 */
/* axidump: post-freeze forensic snapshot (read-only). One cat = every
 * register of interest: PUB AXI bridge debug latches (base
 * 0x300E0000), the PMU domain/sleep/status regs, and the AON DSP boot regs.
 * PUB needs its own mapping (the driver maps it lazily here).
 */
static int f286_axidump(struct f286_radio *rad, char *buf, size_t size)
{
	struct {
		void __iomem *pmu, *aon, *pub;
	} m;
	int n = 0;
	u32 pub_offs[] = { 0x500c, 0x5010, 0x5200, 0x5204, 0x5208, 0x5210 };
	static const char *const pub_names[] = {
		"ADDR_DETECT_CTRL", "ADDR_WRAP_CTRL", "DBG_R_AON",
		"DBG_R_AP",	    "DBG_R_CM4",      "DBG_R_PUBCP"
	};
	unsigned int i;

	m.pmu = rad->pmu;
	m.aon = rad->aon;
	m.pub = ioremap(0x300e0000, 0x4000);

	n += scnprintf(buf + n, size - n, "axidump (u32, reg=value)\n");
	if (m.pmu) {
		static const unsigned int pmu_off[] = { 0x28, 0x2c, 0x34, 0x38,
							0x3c, 0x44, 0x4c, 0x50,
							0xb0, 0xc0, 0xc4, 0xcc,
							0xd4 };
		static const char *const pmu_name[] = { "HU3GE?",
							"PUBCP?",
							"LDSP_CFG",
							"TGDSP_CFG",
							"HU3GE_A",
							"SYS_CFG",
							"?",
							"?",
							"CP_SOFT_RST",
							"PWR_STATUS1_DBG",
							"WTLCP_STATUS",
							"SLEEP_CTRL",
							"WHALE_FSM" };
		for (i = 0; i < ARRAY_SIZE(pmu_off); i++)
			n += scnprintf(buf + n, size - n,
				       "PMU  0x402b%04x %-18s = %08x\n",
				       pmu_off[i], pmu_name[i],
				       readl(m.pmu + pmu_off[i]));
	}
	if (m.aon) {
		static const unsigned int ao[] = { 0x6c,  0x130, 0x144,
						   0x244, 0x248, 0x24c,
						   0x250, 0x254, 0x1144 };
		static const char *const an[] = {
			"AON_REG_PROT",	     "APB_RST2",
			"AUTO_GATE_CTRL1",   "WTL_WCDMA_EB",
			"LDSP_BOOT_VECTOR",  "LDSP_BOOT/EXT_WAIT",
			"TGDSP_BOOT_VECTOR", "TGDSP_BOOT",
			"AUTO_GATE_SET"
		};
		for (i = 0; i < ARRAY_SIZE(ao); i++)
			n += scnprintf(buf + n, size - n,
				       "AON  0x402e%04x %-18s = %08x\n", ao[i],
				       an[i], readl(m.aon + ao[i]));
	}
	if (m.pub) {
		for (i = 0; i < ARRAY_SIZE(pub_offs); i++)
			n += scnprintf(buf + n, size - n,
				       "PUB  0x300e%04x %-18s = %08x\n",
				       pub_offs[i], pub_names[i],
				       readl(m.pub + pub_offs[i]));
		iounmap(m.pub);
	} else {
		n += scnprintf(buf + n, size - n, "PUB  ioremap failed\n");
	}
	/* whale AHB_reg: the t117 FDL1 writes 0x555555 to 0x20E03058, while
	 * another vendor's loader writes the same value to whale-view 0x60830058
	 * -> the AP-visible mirror of whale AHB_reg may be 0x20E03000. Dump
	 * BOTH the suspect mirror and the (reads-as-0) 0x6083 whale window so
	 * the live alias reveals itself.
	 */
	{
		void __iomem *mir;
		static const unsigned int mo[] = { 0x58,   0x74,   0x78,
						   0x7c,   0x1034, 0x1044,
						   0x1060, 0x0044 };
		static const char *const mn[] = {
			"0x58 enable", "0x74 nibbles", "0x78 cfg",
			"0x7c field",  "0x1034 gates", "0x1044 int/module",
			"0x1060 gate", "0x44 bridge"
		};
		mir = ioremap(0x20e03000, 0x2000);
		if (mir) {
			for (i = 0; i < ARRAY_SIZE(mo); i++)
				n += scnprintf(buf + n, size - n,
					       "MIR  0x20e0%04x %-18s = %08x\n",
					       mo[i], mn[i],
					       readl(mir + mo[i]));
			iounmap(mir);
		} else {
			n += scnprintf(buf + n, size - n,
				       "MIR  ioremap failed\n");
		}
		mir = ioremap(0x60830000, 0x2000);
		if (mir) {
			for (i = 0; i < ARRAY_SIZE(mo); i++)
				n += scnprintf(buf + n, size - n,
					       "W6083 0x%04x %-18s = %08x\n",
					       mo[i], mn[i],
					       readl(mir + mo[i]));
			iounmap(mir);
		} else {
			n += scnprintf(buf + n, size - n,
				       "W6083  ioremap failed\n");
		}
	}
	return n;
}

static int f286_queue_send(struct f286_radio *rad, void __iomem *win,
			   u16 cmd_id, const void *payload, u16 payload_len,
			   bool raw8)
{
	u8 body[8 + F286_Q_INLINE_MAX];
	u32 ctl = readl(win + F286_Q_CTL);
	/* in/out come from the DSP-writable shared ctl word. Used unmasked
	 * to index the 8-entry ring, a glitched value >= 0x77F reads/writes
	 * outside the 0x12000 window mapping (kernel oops). Mask both to the
	 * ring modulus before any use; the full-ring check below compares
	 * masked values.
	 */
	u16 raw_in = ctl & 0xffffu;
	u16 raw_out = ctl >> 16;
	u16 in = raw_in & 7;
	u16 out = raw_out & 7;
	u16 next = (in + 1) & 7;
	u16 len; /* LOGICAL size: entry.len + end_off watermark */
	u16 copy_len; /* bytes actually written into the ring body */
	u16 handle = 0; /* split-slot handle */
	bool split;
	u32 body_rel;
	u32 end;

	if (raw_in != in || raw_out != out)
		dev_warn(
			rad->dev,
			"queue: ctl index masked in %u->%u out %u->%u - ring corruption?\n",
			raw_in, in, raw_out, out);

	/* Stock INLINE_MAX / send_message 0x80e03c9a: payloads <= 0x190 go
	 * inline (with a 64-byte cap, cmd 0x74 at 0x148 B and all the big cfg
	 * commands could not be sent); larger payloads take the stock split
	 * path. Split is LTE-only: the stock slot-handle formula (0x811816f0)
	 * is anchored
	 * to the LTE base 0x8073b000.
	 */
	split = !raw8 && payload_len > F286_Q_INLINE_MAX;
	if (payload_len > F286_LTE_SPLIT_SLOT) {
		dev_err(rad->dev, "queue: payload %u too large\n", payload_len);
		return -EINVAL;
	}
	if (split && (win != rad->lte_win || !rad->split_win)) {
		dev_err(rad->dev,
			"queue: payload %u needs the LTE split slots (split_win %p unmapped or not the LTE window)\n",
			payload_len, rad->split_win);
		return -ENOSPC;
	}
	/* Ring-full check (stock put 0x81181edc): `next` becomes
	 * the new `in`, so next==out means all 7 usable entries are in
	 * flight and the DSP has not consumed anything.
	 */
	if (next == out) {
		dev_err(rad->dev,
			"queue: ring FULL (in %u out %u, 7 in flight), EBUSY\n",
			in, out);
		return -EBUSY;
	}
	if (in == out)
		body_rel = F286_Q_BODY;
	else {
		/* Stock alloc_body 0x81181d3c: allocate at slot[in].end_off,
		 * the DSP-consumed watermark (the
		 * slot about to be reused holds the furthest position the DSP
		 * has freed; <=7 in flight keeps it valid). Using
		 * slot[in-1].end_off (the previous entry's own end) instead
		 * collides with undrained traffic under partial drain/wrap.
		 * end_off 0 (wrap marker) lands back at the body-area start.
		 * end_off is DSP-written shared RAM: added to 0xE4E0 with u32
		 * wrap, a bad value would pass the span check and steer the
		 * body memcpy at a wild low window offset. Sane values are only < 0xc80 or the 0 wrap marker
		 * (the AP itself writes nothing else); anything else clamps
		 * to the stock fresh-alloc formula (body-area start).
		 */
		u32 end_off = readl(win + F286_Q_ENTRY + in * 8 + 4);

		if (end_off >= F286_Q_BODY_SPAN && end_off != 0) {
			dev_warn(
				rad->dev,
				"ring: bogus end_off from ring, clamping (slot %u end_off %08x)\n",
				in, end_off);
			end_off = 0;
		}
		body_rel = F286_Q_BODY + end_off;
	}
	/* entry.len carries the LOGICAL message size (payload+8) in all three
	 * shapes (stock put 0x81181edc); only copy_len bytes are written to
	 * the ring body (12 for split: header + u16 handle + pad). The
	 * end_off watermark below advances by the LOGICAL len and wraps to 0
	 * past 0xc7f -- exactly stock's behavior for split sends.
	 */
	if (raw8) {
		memcpy(body, payload, payload_len);
		len = payload_len;
		copy_len = payload_len;
	} else if (split) {
		/* Split slot: round-robin over the 4 x 0x2800 slots. The
		 * mapping is WC (uncached): memcpy_toio stores reach DRAM
		 * coherently for the DSP, no cache clean needed.
		 */
		u8 idx = rad->lte_split_idx & 3u;

		rad->lte_split_idx = (u8)((idx + 1) & 3u);
		handle = idx; /* stock 0x811816f0: (slot - 0x8073b000)/0x2800 */
		memcpy_toio(rad->split_win + (u32)idx * F286_LTE_SPLIT_SLOT,
			    payload, payload_len);
		put_unaligned_le16(cmd_id, body); /* desc[6] u16 @+0 */
		put_unaligned_le16(0xaaaa, body + 2); /* MARKER_SPLIT @+2 */
		/* desc[0] u32 @+4: stock threads the cmdId through send
		 * entry 0x80e0412e; with hdr_layout=1 (default) the id lands
		 * at +4 as well as +0.
		 */
		put_unaligned_le32(hdr_layout ? cmd_id : 0, body + 4);
		put_unaligned_le16(handle, body + 8); /* u16 slot handle */
		put_unaligned_le16(0, body + 10); /* u16 pad (copy 4 B) */
		len = (u16)(payload_len + 8);
		copy_len = 12;
	} else {
		put_unaligned_le16(cmd_id, body); /* desc[6] u16 @+0 */
		put_unaligned_le16(0x5555, body + 2); /* marker @+2 */
		/* desc[0] u32 @+4: stock threads the cmdId through send
		 * entry 0x80e0412e; with hdr_layout=1 (default) the id lands
		 * at +4 as well as +0.
		 */
		put_unaligned_le32(hdr_layout ? cmd_id : 0, body + 4);
		memcpy(body + 8, payload, payload_len);
		len = (u16)(payload_len + 8);
		copy_len = len;
	}
	if (body_rel + copy_len > F286_Q_BODY + F286_Q_BODY_SPAN) {
		dev_err(rad->dev, "queue: body alloc %u@%u out of range\n",
			copy_len, body_rel);
		return -ENOSPC;
	}

	memcpy_toio(win + body_rel, body, copy_len);
	/* Fixed-8 carries the real cmdId in the ring entry; headered sends
	 * use entry.cmdId = 0 and the id rides in the body header.
	 */
	writew(raw8 ? cmd_id : 0, win + F286_Q_ENTRY + in * 8 + 0);
	writew(len, win + F286_Q_ENTRY + in * 8 + 2);
	/* Stock put 0x81181edc: slot[in].end_off gets the raw body start
	 * (0x81181ef4-f04); the final watermark is pre-written into
	 * slot[(in+1)&7].end_off (0x81181f4a-f6e, so stock does touch
	 * entry[next]), then in = next (0x81181f6e). Without that, the next
	 * send's alloc from slot[in] (0x81181d3c) would read a stale
	 * watermark and overlap undrained bodies. The final offset is
	 * reloaded, rounded down to 4 (bic r2,#3), then +4; >= 0xc80 wraps
	 * to 0 (0x81181f64-6c).
	 */
	writel(body_rel - F286_Q_BODY, win + F286_Q_ENTRY + in * 8 + 4);
	end = ((body_rel - F286_Q_BODY + len) & ~3u) + 4;
	if (end >= F286_Q_BODY_SPAN)
		end = 0;
	writel(end, win + F286_Q_ENTRY + next * 8 + 4);
	writew(next, win + F286_Q_CTL);
	dev_info(
		rad->dev,
		"queue: sent cmd %u raw8=%d split=%d handle=%u hdr_layout=%u len %u copy %u body %x start@in %x end prescribed@next %x ctl in %u->%u out %u\n",
		cmd_id, raw8, split, handle, hdr_layout, len, copy_len,
		body_rel, body_rel - F286_Q_BODY, end, in, next, out);
	return 0;
}

/* ---- firmware load (verify-then-touch; hard fail on unmappable target) ---- */

/* Per-step watchdog breadcrumbs: a future block names its step instantly. */
static unsigned int f286_step_no;

static void f286_step_begin(struct f286_radio *rad, const char *name)
{
	f286_step_no++;
	dev_info(rad->dev, "step %u %s start\n", f286_step_no, name);
}

static void f286_step_done(struct f286_radio *rad, const char *name)
{
	dev_info(rad->dev, "step %u %s done\n", f286_step_no, name);
}

/* Direct VFS read from /lib/firmware: the firmware API (usermode helper,
 * compressed-fallback handling) pends forever from kthread context on this
 * platform, so it is bypassed completely.
 *
 * The file is STREAMED in small chunks and never buffered whole: this box
 * has ~32 MB RAM with zram swap, and a whole-file allocation (request_*,
 * or filp_open + one big kernel_read) under memory pressure triggers global
 * reclaim that swaps tmpfs folios to zram - the zram compressor then needs
 * memory to compress into and the writer deadlocks forever in D state
 * (zcomp_stream_get <= zram_submit_bio <= swap_writepage <= shmem_writeout
 * <= shrink_folio_list). If a load wedges anyway, only a reboot helps.
 */
#define F286_FW_CHUNK 0x20000 /* 128 KB staging chunk = max allocation */

static int f286_load_firmware(struct f286_radio *rad, const char *fw_name,
			      unsigned long phys, size_t span)
{
	char path[80];
	struct file *file;
	void *chunk;
	void __iomem *dst;
	bool dst_ioun;
	unsigned int bad = 0, i, chunkno = 0, nchunks;
	loff_t pos = 0;
	ssize_t got;
	size_t off = 0;

	f286_step_begin(rad, "fw-read");
	snprintf(path, sizeof(path), "/lib/firmware/%s", fw_name);
	nchunks = (span + F286_FW_CHUNK - 1) / F286_FW_CHUNK;
	dev_info(
		rad->dev,
		"fw: filp_open %s (%zu bytes expected, %u x %u KB streaming chunks)\n",
		path, span, nchunks, F286_FW_CHUNK / 1024);
	file = filp_open(path, O_RDONLY, 0);
	if (IS_ERR(file)) {
		dev_err(rad->dev, "fw %s: filp_open failed: %d\n", path,
			(int)PTR_ERR(file));
		return PTR_ERR(file);
	}

	/* The ONLY allocation: one 128 KB chunk, __GFP_NORETRY so the alloc
	 * itself can never push reclaim into the zram deadlock above.
	 */
	chunk = kvmalloc(F286_FW_CHUNK, GFP_KERNEL | __GFP_NORETRY);
	if (!chunk) {
		filp_close(file, NULL);
		return -ENOMEM;
	}

	dst = f286_map_dram(rad->dev, phys, span, &dst_ioun);
	if (!dst) {
		kvfree(chunk);
		filp_close(file, NULL);
		return -ENXIO; /* fail hard: the DSP target is unmappable */
	}

	/* Stream: read one chunk, write it straight into the mapped window. */
	while (off < span) {
		size_t want = min_t(size_t, span - off, F286_FW_CHUNK);

		got = kernel_read(file, chunk, want, &pos);
		if (got != want) {
			dev_err(rad->dev,
				"fw %s: short read %zd of %zu at +%zx\n", path,
				got, want, off);
			kvfree(chunk);
			f286_unmap_dram(&dst, dst_ioun);
			filp_close(file, NULL);
			return -EIO;
		}
		memcpy_toio(dst + off, chunk, got);
		off += got;
		chunkno++;
		if ((off & 0xfffff) == 0 || off == span)
			dev_info(rad->dev, "fw: chunk %u/%u (%zu KB)\n",
				 chunkno, nchunks, off >> 10);
	}
	f286_step_done(rad, "fw-read");

	/* Verify sample words in the WINDOW against a fresh positional read
	 * of the file (there is no whole-file buffer to compare with).
	 */
	f286_step_begin(rad, "window-copy");
	for (i = 0; i < span / 4; i += 997) {
		u32 rb = readl(dst + i * 4);
		loff_t fpos = (loff_t)i * 4;

		if (kernel_read(file, chunk, 4, &fpos) == 4 &&
		    rb != get_unaligned_le32(chunk))
			bad++;
	}
	dev_info(
		rad->dev,
		"firmware %s: %zu bytes -> %08lx, sample mismatches %u, head %08x %08x\n",
		fw_name, span, phys, bad, readl(dst), readl(dst + 4));

	/* Coherence check: the sample verify above re-reads the
	 * SAME WC mapping, so a silently-cached staging path would verify
	 * clean while the DSP reads stale DRAM. Map the same physical range
	 * a second time with CACHEABLE attributes through an INDEPENDENT VA
	 * (cache lines are VA-tagged: dirty lines from a cached staging
	 * write under the first mapping are invisible to this one) and read
	 * the head words. Identical -> "fw coherent". Different -> the
	 * staging write sat in cache: clean the staging range (by-VA dcache
	 * clean of the mapping the copy used) so it reaches DRAM before the
	 * DSP release. memremap(MEMREMAP_WB) is NOT used: this region has no
	 * struct pages (below the DT memory node), WB would fall back to the
	 * same WC/uncached alias we already have - ioremap_cache is the
	 * only genuinely cacheable mapping available here.
	 */
	{
		u32 wc0 = readl(dst), wc1 = readl(dst + 4);
		void __iomem *cmap = ioremap_cache(phys, span);

		if (cmap) {
			u32 ca0 = readl(cmap), ca1 = readl(cmap + 4);

			dev_info(rad->dev,
				 "fw verify: wc=%08x %08x cached=%08x %08x\n",
				 wc0, wc1, ca0, ca1);
			if (ca0 == wc0 && ca1 == wc1) {
				dev_info(rad->dev, "fw coherent\n");
			} else {
				dev_err(rad->dev,
					"CACHE COHERENCE BUG CONFIRMED - cleaning staging range before DSP release\n");
				clean_dcache_area((__force void *)dst, span);
			}
			iounmap(cmap);
		} else {
			dev_warn(
				rad->dev,
				"fw verify: cached map unavailable, wc=%08x %08x only\n",
				wc0, wc1);
		}
	}
	/*
	 * Clean the staged image unconditionally before the DSP is
	 * released (inferred, not seen in stock code), not only when the
	 * cached-alias probe above catches a mismatch. A no-op for a
	 * genuinely WC mapping, a real fix for any fallback cached alias.
	 */
	clean_dcache_area((__force void *)dst, span);
	dev_info(
		rad->dev,
		"fw: unconditional dcache clean %zx bytes before DSP release\n",
		span);
	kvfree(chunk);
	f286_unmap_dram(&dst, dst_ioun);
	filp_close(file, NULL);
	f286_step_done(rad, "window-copy");
	return 0;
}

/* Stock L1 shared-memory init before the boot (0x80e4c6e4 -> 0x80eee990):
 * zero win+0 (0x9cc) and [0x81712bac+0x18] (0xb2c; 0x800ff300 in the
 * emulator, inferred fixed); the IRAM part only with gsm_iram.
 */
static void f286_gsm_l1_init(struct f286_radio *rad, void __iomem *g,
			     void __iomem *iram)
{
	memset_io(g, 0, 0x9cc);
	memset_io(g + 0x1f300, 0, 0xb2c);
	if (iram) {
		memset_io(iram + 0xc00, 0, 0xe00);
		writel(1, iram + 0xc00 + 0xdcc);
	}
	dev_info(rad->dev,
		 "gsm: L1 init clears 0x800e0000+9cc, 0x800ff300+b2c%s\n",
		 iram ? ", IRAM 0x10c00+e00 ([+dcc]=1)" :
			" (IRAM skipped, gsm_iram=0)");
}

static void f286_gsm_params(struct f286_radio *rad, void __iomem *g,
			    void __iomem *iram)
{
	/* GSM DSP pre-release param download, stock 0x810e910c. g maps the
	 * window base 0x800e0000 ([0x81712f18]); every offset below is the
	 * stock one relative to it (the old code applied them to the image
	 * base 0x80100000 and overwrote DSP code). Cell values: from a
	 * running stock phone's memory (pre-release values
	 * inferred). The header word (1) is written LAST.
	 */
	void __iomem *p = g + F286_GSM_PARAMS;
	unsigned int i;
	u32 rb = 0;

	writel(0, g + 0x20044); /* image header +0x44 (ready word) */
	writel(0, g + 0xb8000);
	writel(0xbb, g + 0x20048); /* image header +0x48; 0xaa needs NV cfg */
	memset_io(p, 0, 0x6f8);
	writel(0, p + 0x638); /* 0x80a06df0() = constant 0 */
	writel(0, p + 0x63c);
	writel(1, p + 0x6ec); /* NV 0x1b1 check, live 1 */
	writel(0, p + 0x6f4);
	writel(1, p + 0x774); /* [0x8170204c+0x10], live 1 */
	writel(0x00010003, p + 0x6e4); /* 0x80a080f2(2), live 0x00010003 */
	writel(1, p); /* header word = 1 (LAST) */
	memset_io(g + F286_GSM_AUDIO, 0, 0xb0ac); /* 0x80b1c610 */
	if (iram) {
		/* 0x80b18218: mode cell, read back until it sticks (7 tries) */
		for (i = 0; i < 7; i++) {
			writel(1, iram + 0x1d30);
			rb = readl(iram + 0x1d30);
			if (rb == 1)
				break;
		}
	}
	dev_info(
		rad->dev,
		"gsm params @0x800f9670: 0x20044/0xb8000=0, 0x20048=0xbb, +638=0 +6ec=1 +774=1 +6e4=10003, params[0]=1 LAST, audio 0x80388000+b0ac cleared, mode cell %s\n",
		iram ? (rb == 1 ? "0x11d30=1" : "0x11d30 DID NOT STICK") :
		       "skipped (gsm_iram=0)");
}

/* Stock sharedmem_proc 0x80ef4512: poll the run word's low u16 (one OS
 * tick per try, assert after 5000), then require rf_open/rf_close != 0.
 */
static void f286_gsm_run_check(struct f286_radio *rad, void __iomem *g)
{
	unsigned int tries;
	u16 run = 0;

	for (tries = 0; tries < 5000; tries++) {
		run = readw(g + F286_GSM_RUN_WORD);
		if (run)
			break;
		mutex_unlock(&rad->lock);
		msleep_interruptible(1);
		mutex_lock(&rad->lock);
	}
	dev_info(
		rad->dev,
		"gsm: run word 0x800f9ca4 = %04x after %u polls (%s); rf_open %04x rf_close %04x (live stock 0133 / 0087)\n",
		run, tries + 1,
		run ? "DSP RUNNING" : "NOT RUN, stock asserts here",
		readw(g + F286_GSM_RF_OPEN), readw(g + F286_GSM_RF_CLOSE));
}

/* ---- XDSP clock-on (the AP-executable part of the stock sequence) ----
 * Suspected cause of the silent DSPs at the time this was written: the DSP
 * exec clock TWPLL_51M2 (51.2 MHz) was never enabled. Run from both load threads between power verification
 * and the DSP release, in stock order: light-path trio first (it precedes
 * every 0xD15E access in all three stock power-on fns), whale module
 * enables, TWPLL DSP-branch ops, then the die-local release attempt.
 */

/* Guarded die-local clock release: own fire-and-forget kthread. Addresses
 * are physical (the stock MMU maps them 1:1): whale AON 0xD15E0000,
 * RF gate 0xD15D0000 (the old 0x515E/0x515D probes aborted because those are
 * simply not the whale bus addresses). Sequence = stock clock-config block
 * (ensure-on 0x80D0F2FA chain): read +0x28 (stock's own first touch),
 * SET bank bits 9:7, CLEAR bank bits 9:7 then bit21 (clock release), the
 * 0x60831034=0x410 per-poweron cfg, whale AON enable |1, RF gate =2. A death
 * here still lets the main load continue after a 100 ms wait regardless.
 */
static int f286_dieclk_thread(void *data)
{
	struct f286_radio *rad = data;
	void __iomem *e, *d, *apb;
	u32 v;

	/* This thread maps its own private copies of the
	 * die-local + whale pages and holds them for its whole lifetime;
	 * it NEVER dereferences the rad-> VAs, so f286_radio_remove can
	 * unmap the shared maps at any time without a UAF. A thread wedged
	 * on a dead-bus read (its designed failure mode) leaks only these
	 * private mappings until reboot; a one-shot leak beats unmapping
	 * underneath a live thread.
	 */
	e = ioremap(F286_WAON_E_PHYS, F286_WAON_E_SPAN);
	d = ioremap(F286_WAON_D_PHYS, F286_WAON_D_SPAN);
	apb = ioremap(F286_WHALE_APB_PHYS, F286_WHALE_APB_SPAN);
	if (!e || !d || !apb) {
		dev_err(rad->dev,
			"dieclk: private map failed (e %p d %p apb %p), aborting\n",
			e, d, apb);
		if (e)
			iounmap(e);
		if (d)
			iounmap(d);
		if (apb)
			iounmap(apb);
		rad->dieclk_thread = NULL;
		return -ENXIO;
	}

	dev_info(
		rad->dev,
		"dieclk: pre-probe READ 0xD15E0028 (die powered - should answer)\n");
	v = readl(e + 0x028);
	dev_info(rad->dev, "dieclk: 0xD15E0028 = %08x (read survived)\n", v);
	/* model/whale.cpp light_path: the die-local trio tail [0xD15E0068] =
	 * 0x00DD00DD, the first write of the guarded thread (0xE010100C/
	 * 1014 AP-side twins are opt-in behind e010_light in xdsp_clock_on).
	 */
	writel(0x00dd00dd, e + 0x068);
	dev_info(rad->dev,
		 "dieclk: 0xD15E0068 <- 00dd00dd (light-path tail), rb %08x\n",
		 readl(e + 0x068));
	writel(0x380, e + 0x1028); /* SET bank of +0x28: bits 9:7 */
	dev_info(rad->dev,
		 "dieclk: 0xD15E1028 <- 380 (SET b9:7), +0x28 rb %08x\n",
		 readl(e + 0x028));
	writel(0x380, e + 0x2000); /* CLEAR bank of +0x00: bits 9:7 */
	dev_info(rad->dev,
		 "dieclk: 0xD15E2000 <- 380 (CLR b9:7), +0x00 rb %08x\n",
		 readl(e));
	writel(0x300000, e + 0x2000); /* CLEAR bank of +0x00: bit21 (release) */
	dev_info(
		rad->dev,
		"dieclk: 0xD15E2000 <- 300000 (CLR b21 release), 0xD15E0000 rb %08x\n",
		readl(e));
	writel(0x410, apb + 0x1034); /* 0x60831034 (private map) */
	dev_info(rad->dev, "dieclk: 0x60831034 <- 410, rb %08x\n",
		 readl(apb + 0x1034));
	v = readl(e + 0x058);
	writel(v | 1, e + 0x058);
	dev_info(rad->dev, "dieclk: 0xD15E0058 %08x -> |1, rb %08x\n", v,
		 readl(e + 0x058));
	v = readl(d + 0x020);
	writel(0, d + 0x020);
	dev_info(rad->dev, "dieclk: 0xD15D0020 %08x -> 0 (rb %08x)\n", v,
		 readl(d + 0x020));
	writel(2, d + 0x020);
	dev_info(rad->dev, "dieclk: 0xD15D0020 -> 2 (rb %08x)\n",
		 readl(d + 0x020));
	iounmap(e);
	iounmap(d);
	iounmap(apb);
	dev_info(rad->dev,
		 "dieclk: guarded sequence complete, private maps released\n");
	/* Clear the alive guard before returning; a thread wedged
	 * mid-sequence on a dead-bus read never gets here and remove
	 * proceeds regardless (only the private maps leak).
	 */
	rad->dieclk_thread = NULL;
	return 0;
}

static void f286_xdsp_clock_on(struct f286_radio *rad)
{
	u32 v;

	/* 1. light path: whale misc die cfg 0xE0100000 (physical; the
	 * 0x6010102C page tried earlier is a different block). 0xE010100C
	 * clr-bit24 then 0xE0101014=0xCC is the stock ensure-on trio tail and
	 * precedes every 0xD15E access. Opt-in (e010_light, default off):
	 * this page hung the bus on hardware, and a dead-bus read here leaves
	 * the load kthread in D state holding rad->lock, after which every
	 * sysfs access piles up behind it.
	 */
	if (!e010_light) {
		dev_info(
			rad->dev,
			"e010 light: skipped (param off) - 0xE010100C/1014 not touched (bus-hang class); echo 1 > /sys/module/f286_radio/parameters/e010_light to enable\n");
	} else if (f286_get_mmio_locked(rad, &rad->wcfg_e010, F286_WCFG_PHYS,
					F286_WCFG_SPAN, "wcfg-0xe010") == 0) {
		v = readl(rad->wcfg_e010 + 0x100c);
		writel(v & ~0x1000000u, rad->wcfg_e010 + 0x100c);
		dev_info(rad->dev,
			 "xdspclk: 0xE010100C %08x -> %08x (clr 01000000)\n", v,
			 readl(rad->wcfg_e010 + 0x100c));
		writel(0xcc, rad->wcfg_e010 + 0x1014);
		dev_info(rad->dev, "xdspclk: 0xE0101014 <- cc, rb %08x\n",
			 readl(rad->wcfg_e010 + 0x1014));
	}

	/* 2. whale module enable (whale-APB, AP-writable). */
	if (f286_get_mmio_locked(rad, &rad->whale_apb, F286_WHALE_APB_PHYS,
				 F286_WHALE_APB_SPAN, "whale-0x6083") == 0) {
		writel(0x80, rad->whale_apb + 0x1044); /* 0x60831044 */
		dev_info(rad->dev, "xdspclk: 0x60831044 <- 80, rb %08x\n",
			 readl(rad->whale_apb + 0x1044));

		/* 3. TWPLL_51M2 enable (AP clock mgr 0x402D0000 enable-op
		 * tails: 0x60831034 = 0x40000, 0x60831060 = 0x2000).
		 * No "twin" writes to 0x60832034/0x60832060: whale-APB has
		 * SET/CLEAR banks, so the +0x2000 aliases most likely cancel
		 * the +0x0000 sets instead of making them stick.
		 */
		writel(0x40000, rad->whale_apb + 0x1034); /* 0x60831034 */
		dev_info(rad->dev, "xdspclk: 0x60831034 <- 40000, rb %08x\n",
			 readl(rad->whale_apb + 0x1034));
		writel(0x2000, rad->whale_apb + 0x1060); /* 0x60831060 */
		dev_info(rad->dev, "xdspclk: 0x60831060 <- 2000, rb %08x\n",
			 readl(rad->whale_apb + 0x1060));

		/* 4. whale enable (0x80D0E90C = [0x60830044] |= 0x80 +
		 * spin(0x50), at the short offset as in stock).
		 */
		f286_rmw32(rad, "xdspclk 0x60830044", rad->whale_apb, 0x044,
			   0x80, 0);
		f286_whale_settle(rad);
	}

	/* 4b. XDSP exec-clock source mux (model/clocks.h
	 * xdsp_clk_mux_select): 0x402D02B4 sel 2 = TWPLL_51M2. The only
	 * 0x402D0000 mapping in this driver; lazy-mapped, a failure logs and
	 * continues.
	 */
	if (xdsp_mux > 3) {
		dev_info(rad->dev,
			 "xdspclk: 0x402D02B4 mux SKIPPED (xdsp_mux=%u > 3)\n",
			 xdsp_mux);
	} else if (f286_get_mmio_locked(rad, &rad->clkm, F286_CLKM_PHYS,
					F286_CLKM_SPAN, "clkm-0x402d") == 0) {
		u32 mux = readl(rad->clkm + 0x02b4);

		if ((mux & 3u) == xdsp_mux) {
			dev_info(
				rad->dev,
				"xdspclk: 0x402D02B4 = %08x (sel %u TWPLL_51M2 already set)\n",
				mux, xdsp_mux);
		} else {
			writel((mux & ~3u) | xdsp_mux, rad->clkm + 0x02b4);
			dev_info(
				rad->dev,
				"xdspclk: 0x402D02B4 = %08x -> %08x (sel TWPLL_51M2), rb %08x\n",
				mux, (mux & ~3u) | xdsp_mux,
				readl(rad->clkm + 0x02b4));
		}
	}

	/* 5. die-local clock release attempt, opt-in (see dieclk above).
	 * Default off: the pre-probe read of 0xD15E0028 hung the bus on
	 * hardware even with the die powered (same interconnect-stall
	 * signature as the old 0x515e attempts); the whale AON block is not
	 * AP-reachable in our state for a reason not yet found. Set
	 * /sys/module/f286_radio/parameters/dieclk to 1 BEFORE load_lte to
	 * attempt it; the default path stays clean.
	 */
	if (!dieclk) {
		dev_info(
			rad->dev,
			"dieclk: skipped (param off, default) - die-local exec-clock release 0xD15E/0xD15D not attempted; echo 1 > /sys/module/f286_radio/parameters/dieclk before load_lte to try it (bus-hang class, guarded kthread only)\n");
	} else if (f286_get_mmio_locked(rad, &rad->waon_e, F286_WAON_E_PHYS,
					F286_WAON_E_SPAN, "waon-0xd15e") == 0 &&
		   f286_get_mmio_locked(rad, &rad->waon_d, F286_WAON_D_PHYS,
					F286_WAON_D_SPAN, "waon-0xd15d") == 0 &&
		   rad->whale_apb) {
		struct task_struct *t;

		dev_info(
			rad->dev,
			"xdspclk: dieclk=1, spawning guarded die-clock kthread (fire-and-forget, own private maps, load continues +100 ms)\n");
		t = kthread_run(f286_dieclk_thread, rad, "f286-dieclk");
		if (IS_ERR(t)) {
			dev_warn(rad->dev,
				 "xdspclk: dieclk kthread spawn failed %pe\n",
				 t);
		} else {
			/* Publish the alive guard (bookkeeping only: the
			 * thread maps its own private copies, so remove
			 * can unmap the shared VAs at any time).
			 */
			rad->dieclk_thread = t;
			/* Sleep with the lock released. */
			mutex_unlock(&rad->lock);
			msleep_interruptible(100);
			mutex_lock(&rad->lock);
			dev_info(
				rad->dev,
				"xdspclk: +100 ms, load continues (dieclk fate independent)\n");
		}
	}

	/* 6. WCDMA clock force-enable + AON_REG_PROT key re-assert (0x9620
	 * is the protected-LDSP-control unlock key, not a clock enable; stock
	 * HAL 0x811d0db4 writes it before touching the DSP boot regs).
	 * 0x402e1144 = AUTO_GATE_CTRL1 SET: 0x1a = b1/b3 WCDMA wtlcp/pubcp
	 * force_en + b4 root auto-gate sel. Stock's tail also writes
	 * [0x60831044] = 0x40 (whale DSP int/module enable) right after the
	 * 0x1144 = 0x1a; without it the 0x80 from step 2 stays there. The
	 * 0x40 lands after the dieclk decision above.
	 */
	writel(0x1a, rad->aon + F286_AON_WCDMA_FORCE);
	dev_info(rad->dev,
		 "xdspclk: 0x402e1144 <- 1a (AUTO_GATE_CTRL1 SET), rb %08x\n",
		 readl(rad->aon + F286_AON_WCDMA_FORCE));
	if (rad->whale_apb) {
		writel(0x40, rad->whale_apb + 0x1044); /* 0x60831044 */
		dev_info(
			rad->dev,
			"xdspclk: 0x60831044 <- 40 (stock int-enable tail, was 80), rb %08x\n",
			readl(rad->whale_apb + 0x1044));
	}
	f286_rmw32(rad, "xdspclk 0x402e006c", rad->aon, F286_AON_CLK_EN, 0x9620,
		   0);
}

/* Reset release is an AON request with a visible readback. Vendor code
 * for a sibling chip treats a non-zero readback as "still in reset" and
 * re-writes the clear on its resume path. Read the release
 * register back; if the bit still reads set, retry once after 10 ms and
 * log the outcome.
 */
static void f286_release_readback(struct f286_radio *rad, u32 bit,
				  const char *who)
{
	u32 rb = readl(rad->aon + F286_AON_DSP_RESET);
	u32 w;

	dev_info(rad->dev, "%s release readback: 0x402e0130=%08x bit%u %s\n",
		 who, rb, __ffs(bit), (rb & bit) ? "STILL SET" : "clear");
	if (!(rb & bit))
		return;
	/* Sleep with the lock released (called from locked context). */
	mutex_unlock(&rad->lock);
	usleep_range(10000, 20000);
	mutex_lock(&rad->lock);
	w = rb & ~bit;
	writel(w, rad->aon + F286_AON_DSP_RESET);
	rb = readl(rad->aon + F286_AON_DSP_RESET);
	dev_info(rad->dev,
		 "%s release retry +10ms: wrote %08x rb %08x bit%u %s\n", who,
		 w, rb, __ffs(bit), (rb & bit) ? "STILL SET" : "clear");
}

static int f286_load_gsm_locked(struct f286_radio *rad)
{
	void __iomem *g; /* 0x800e0000 window base .. image end */
	void __iomem *iram = NULL;
	bool g_ioun;
	unsigned int i;
	int ret;

	ret = f286_get_mmio_locked(rad, &rad->pmu, F286_PMU_PHYS, F286_PMU_SPAN,
				   "pmu-0x402b");
	if (!ret)
		ret = f286_get_mmio_locked(rad, &rad->aon, F286_AON_PHYS,
					   F286_AON_SPAN, "aon-0x402e");
	if (!ret)
		ret = f286_get_dram_locked(rad, &rad->gsm_win,
					   &rad->gsm_win_ioun,
					   F286_GSM_WIN_PHYS, F286_WIN_SPAN,
					   "gsm-window-0x800e0000");
	if (ret)
		return ret;
	if (stage) {
		dev_info(
			rad->dev,
			"gsm: stage=1 -> window mapped (gsm_win=%p), stopping before firmware load\n",
			rad->gsm_win);
		return 0;
	}

	ret = f286_load_firmware(rad, rad->fw_names[1], F286_GSM_DRAM_PHYS,
				 F286_GSM_DRAM_SPAN);
	if (ret)
		return ret;
	g = f286_map_dram(rad->dev, F286_GSM_WIN_PHYS, F286_GSM_BASE_SPAN,
			  &g_ioun);
	if (!g)
		return -ENXIO;
	if (gsm_iram) {
		iram = ioremap(F286_GSM_IRAM_PHYS, F286_GSM_IRAM_SPAN);
		if (!iram)
			dev_warn(
				rad->dev,
				"gsm: IRAM %08lx unmappable, IRAM cells skipped\n",
				F286_GSM_IRAM_PHYS);
	}
	f286_gsm_l1_init(rad, g, iram);

	/* Format the GSM command ring before release (same layout, ctl at
	 * win+0xe400; the GSM DSP polls it like its LTE sibling).
	 */
	writel(0, rad->gsm_win + F286_Q_CTL);
	for (i = 0; i < 8 * 8; i += 4)
		writel(0, rad->gsm_win + F286_Q_ENTRY + i);
	writel(0, rad->gsm_win + F286_Q_BODY);
	writel(0, rad->gsm_win + F286_Q_BODY + 4);
	dev_info(
		rad->dev,
		"gsm: queue ctl initialized (in=0 out=0, readback %08x), entries+body zeroed\n",
		readl(rad->gsm_win + F286_Q_CTL));
	dev_info(rad->dev, "gsm: hold reset (0x402e0130 bit17)\n");
	f286_rmw32(rad, "aon+130", rad->aon, F286_AON_DSP_RESET, 0x20000, 0);
	f286_gsm_params(rad, g, iram);
	f286_rmw32(rad, "aon+254", rad->aon, F286_AON_GSM_START, 0, 1);
	f286_rmw32(rad, "aon+130", rad->aon, F286_AON_DSP_RESET, 0x20000, 0);
	f286_rmw32(rad, "aon+06c", rad->aon, F286_AON_CLK_EN, 0x9620, 0);
	/* GSM release (model/pmu.cpp xdsp_power_release_wait(2),
	 * model/boot_hal.h release_power): exactly two CLR-bank writes,
	 * 0x402b2038 = 0x03000000 (clear PD_WTLCP_TGDSP_CFG b25:24, the
	 * GSM/TD DSP domain) and 0x402b2044 = 0x03000000 (clear
	 * PD_WTLCP_SYS_CFG b25:24; 0x44 b25 FORCE_SHUTDOWN resets to 1, so the
	 * WTLCP_SYS release is required here). 0x402b2044 = 0x30000 would
	 * clear a different sub-field (0044 b17:16) that no stock code
	 * touches.
	 */
	dev_info(rad->dev, "gsm: 0x402b0038=%08x 0x402b0044=%08x (pre)\n",
		 readl(rad->pmu + 0x0038), readl(rad->pmu + F286_PMU_RD_0044));
	/* Defensive re-clear: SLEEP_CTRL b17 (WTLCP_FORCE_DEEP_SLEEP) resets
	 * to 1 (0x402b00cc reset value 0x00220000). This path normally relies on radio_power having run
	 * first; if load_gsm is ever used standalone (or after a warm restart
	 * that re-armed b17), the TGDSP domain would boot force-slept. Cheap
	 * insurance, idempotent with the radio_power step.
	 */
	writel(0x00020000, rad->pmu + F286_PMU_CLR_DSP_CFG);
	writel(0x03000000, rad->pmu + F286_PMU_CLR_2038); /* clr 0038 b25:24 */
	writel(0x03000000, rad->pmu + F286_PMU_CLR_2044); /* clr 0044 b25:24 */
	dev_info(
		rad->dev,
		"gsm: 0x402b2038 <- 03000000 (0038=%08x rb), 0x402b2044 <- 03000000 (0044=%08x rb)\n",
		readl(rad->pmu + 0x0038), readl(rad->pmu + F286_PMU_RD_0044));
	/* Stock (model/pmu.cpp xdsp_power_release_wait(2)) polls STATUS0 &
	 * 0x7c00 == 0, the [14:10] DSP-domain power field, not WHu3g c0[4:0]
	 * (the island radio_power already drives). Naming note: the register
	 * manual calls [14:10] PD_WTLCP_LDSP_STATE and [9:5] TGDSP; some log
	 * lines and the model call them XDSP-B / XDSP-A, which is the other
	 * way round. The 0x7c00 mask follows the stock code (DSP id 2).
	 */
	dev_info(rad->dev, "gsm: cores poll: c0 & 0x7c00 -> 0\n");
	{
		unsigned int tries;
		bool b_on = false;

		for (tries = 0; tries < 200; tries++) {
			u32 c0v = readl(rad->pmu + F286_PMU_POWER_STATUS);

			b_on = ((c0v & 0x7c00) == 0);
			dev_info(rad->dev,
				 "gsm: xdsp poll %u: c0=%08x B[14:10]=%u\n",
				 tries, c0v, (c0v >> 10) & 0x1f);
			if (b_on)
				break;
			/* Sleep with the lock released (up to 3 s of
			 * polling must not pin rad->lock).
			 */
			mutex_unlock(&rad->lock);
			msleep_interruptible(1);
			mutex_lock(&rad->lock);
		}
		dev_info(rad->dev,
			 "gsm: xdsp B[14:10] %s after %u samples (c0=%08x)\n",
			 b_on ? "ON (power released)" : "STILL HELD after 3 s",
			 tries + 1, readl(rad->pmu + F286_PMU_POWER_STATUS));
	}
	dev_info(
		rad->dev,
		"gsm: boot-addr 0x402e0250 = %08x (mode %u: 0=corrected>>1 0x40080040, 1=raw 0x80100080, 2=corrected+4; entry 0x80100080)\n",
		f286_gsm_boot_addr(), boot_addr_mode);
	f286_step_begin(rad, "boot-addr");
	writel(f286_gsm_boot_addr(), rad->aon + F286_AON_GSM_BOOT_ADDR);
	dev_info(rad->dev, "gsm: boot-addr readback %08x\n",
		 readl(rad->aon + F286_AON_GSM_BOOT_ADDR));
	f286_step_done(rad, "boot-addr");
	udelay(100);
	/* Stock GSM boot 0x810b33fe sets 0x402e0254 bit0 (GSM_START_CTRL,
	 * HAL 0x811d0d4e), not the LTE aon+24c bit.
	 */
	dev_info(rad->dev, "gsm: start-ctrl high (bit0, HAL 0x811d0d4e)\n");
	f286_step_begin(rad, "start-ctrl");
	f286_rmw32(rad, "aon+254", rad->aon, F286_AON_GSM_START, 1, 0);
	f286_step_done(rad, "start-ctrl");
	f286_xdsp_clock_on(rad); /* XDSP clock-on (TWPLL_51M2) before release */
	f286_step_begin(rad, "release");
	f286_rmw32(rad, "aon+130", rad->aon, F286_AON_DSP_RESET, 0,
		   0x20000); /* RELEASE */
	f286_release_readback(rad, 0x20000, "gsm"); /* bit17 stuck check */
	f286_step_done(rad, "release");
	dev_info(rad->dev,
		 "gsm: released, entry 0x80100080, queue 0x800ee400\n");

	/* Stock success criterion: run word, then rf_open/rf_close (up to
	 * ~5 s, lock released while sleeping). Then the power-status
	 * evidence (c0 [9:5]/[14:10] = XDSP-A/B).
	 */
	f286_gsm_run_check(rad, g);
	{
		u32 c0 = readl(rad->pmu + F286_PMU_POWER_STATUS);
		u32 c4 = readl(rad->pmu + F286_PMU_WTLCP_STATUS);
		u32 d4 = readl(rad->pmu + F286_PMU_WHALE_FSM);

		dev_info(
			rad->dev,
			"gsm: post-release c0=%08x (XDSP-A[9:5]=%u XDSP-B[14:10]=%u) c4=%08x (WTLCP[4:0]=%u) d4=%08x (state=%u)\n",
			c0, (c0 >> 5) & 0x1f, (c0 >> 10) & 0x1f, c4, c4 & 0x1f,
			d4, (d4 >> 4) & 0xf);
	}

	if (iram)
		iounmap(iram);
	f286_unmap_dram(&g, g_ioun);
	return 0;
}

/* ---- LTE XDSP cores A/B (stock 0x80d2d5d0 core A, 0x80d2dcc4 core B) ----
 * Per core: param pulse; ready = 0; zero win+0x40000/+0x48000 (0x7ff words
 * each); copy the boot loader to win; pulse; wait ready == 1; param pulse;
 * copy the main image; pulse; wait ready == 2. Stock delays 4 chip units
 * (~1 us, 0x80d1f812) before each ctl write except the 0x1e burst, and
 * spins forever on the ready word; here the wait is bounded. Image word
 * counts come from the stock RW data (0x81713f38..4c, read at runtime in
 * the emulator): A boot 0x696 / main 0x26ad, B boot 0x556 / main 0x2627.
 */
struct f286_xdsp_core {
	const char *name;
	u32 win, ctl, ready; /* offsets from F286_XDSP_PHYS */
	const char *boot, *main;
	size_t boot_len, main_len;
	bool late_param; /* core B: 0,1,0 before the param word */
};

static const struct f286_xdsp_core f286_xdsp_core_tab[] = {
	{ "A", 0x00000, 0x7f000, 0x19908, "ums9117/f286-xdspA-boot.bin",
	  "ums9117/f286-xdspA-main.bin", 0x696 * 4, 0x26ad * 4, false },
	{ "B", 0x80000, 0xff000, 0x99540, "ums9117/f286-xdspB-boot.bin",
	  "ums9117/f286-xdspB-main.bin", 0x556 * 4, 0x2627 * 4, true },
};

#define F286_XDSP_TICK_US 2

static void f286_xdsp_ctl(void __iomem *ctl, u32 v)
{
	udelay(F286_XDSP_TICK_US);
	writel(v, ctl);
}

static void f286_xdsp_burst(void __iomem *ctl)
{
	writel(0x1e, ctl);
	writel(0x1e, ctl);
	writel(0x1e, ctl);
}

/* 0,1,[param],0,0x1e (A, and the opening pulse of both); B's second param
 * pulse is 0,1,0,[param],0,0x1e.
 */
static void f286_xdsp_param_pulse(void __iomem *win, void __iomem *ctl,
				  bool late)
{
	f286_xdsp_ctl(ctl, 0);
	f286_xdsp_ctl(ctl, 1);
	if (late)
		f286_xdsp_ctl(ctl, 0);
	writel(F286_XDSP_PARAM, win + 0x40);
	f286_xdsp_ctl(ctl, 0);
	f286_xdsp_ctl(ctl, 0x1e);
}

static void f286_xdsp_run_pulse(void __iomem *ctl)
{
	f286_xdsp_ctl(ctl, 0);
	f286_xdsp_ctl(ctl, 1);
	f286_xdsp_ctl(ctl, 0);
	udelay(F286_XDSP_TICK_US);
	f286_xdsp_burst(ctl);
}

/* The device-data capsule ships these small images as .xz; the firmware
 * loader (FW_LOADER_COMPRESS_XZ) decompresses them, filp_open would not.
 */
static int f286_xdsp_read_img(struct f286_radio *rad, const char *fw_name,
			      size_t want, void *buf)
{
	const struct firmware *fw;
	int ret;

	ret = request_firmware_direct(&fw, fw_name, rad->dev);
	if (ret) {
		dev_err(rad->dev, "xdsp: %s: request_firmware failed: %d\n",
			fw_name, ret);
		return ret;
	}
	if (fw->size != want) {
		dev_err(rad->dev, "xdsp: %s: %zu bytes, expected %zu\n",
			fw_name, fw->size, want);
		ret = -EIO;
	} else {
		memcpy(buf, fw->data, want);
	}
	release_firmware(fw);
	return ret;
}

static void f286_xdsp_copy(void __iomem *dst, const void *src, size_t len)
{
	size_t i;

	for (i = 0; i < len; i += 4)
		writel_relaxed(get_unaligned_le32(src + i), dst + i);
	/* Order the relaxed image writes before the caller releases the DSP. */
	wmb();
}

static int f286_xdsp_wait_ready(struct f286_radio *rad,
				const struct f286_xdsp_core *c,
				void __iomem *ready, u32 want)
{
	u32 v;
	int ret;

	ret = readl_poll_timeout(ready, v, v == want, 10, 200000);
	dev_info(rad->dev, "xdsp %s: ready[%08lx] = %x (want %x)%s\n", c->name,
		 F286_XDSP_PHYS + c->ready, v, want,
		 ret ? " TIMEOUT 200 ms" : "");
	return ret;
}

static int f286_xdsp_boot_core(struct f286_radio *rad, void __iomem *base,
			       const struct f286_xdsp_core *c, void *img)
{
	void __iomem *win = base + c->win;
	void __iomem *ctl = base + c->ctl;
	void __iomem *ready = base + c->ready;
	unsigned int i;
	int ret;

	dev_info(rad->dev, "xdsp %s: ctl %08x ready %08x (pre)\n", c->name,
		 readl(ctl), readl(ready));
	f286_xdsp_param_pulse(win, ctl, false);
	writel(0, ready);
	for (i = 0; i < 0x7ff; i++) {
		writel_relaxed(0, win + 0x40000 + i * 4);
		writel_relaxed(0, win + 0x48000 + i * 4);
	}

	ret = f286_xdsp_read_img(rad, c->boot, c->boot_len, img);
	if (ret)
		return ret;
	f286_xdsp_copy(win, img, c->boot_len);
	dev_info(rad->dev, "xdsp %s: boot loader %zx bytes, head rb %08x\n",
		 c->name, c->boot_len, readl(win));
	f286_xdsp_run_pulse(ctl);
	ret = f286_xdsp_wait_ready(rad, c, ready, 1);
	if (ret)
		return ret;

	f286_xdsp_param_pulse(win, ctl, c->late_param);
	ret = f286_xdsp_read_img(rad, c->main, c->main_len, img);
	if (ret)
		return ret;
	f286_xdsp_copy(win, img, c->main_len);
	dev_info(rad->dev, "xdsp %s: main %zx bytes, head rb %08x\n", c->name,
		 c->main_len, readl(win));
	f286_xdsp_run_pulse(ctl);
	return f286_xdsp_wait_ready(rad, c, ready, 2);
}

static int f286_xdsp_boot_cores(struct f286_radio *rad)
{
	void __iomem *base;
	void *img;
	unsigned int i;
	int ret = 0;

	img = kvmalloc(F286_XDSP_IMG_MAX, GFP_KERNEL | __GFP_NORETRY);
	if (!img)
		return -ENOMEM;
	base = ioremap(F286_XDSP_PHYS, F286_XDSP_SPAN);
	if (!base) {
		kvfree(img);
		dev_err(rad->dev, "xdsp: ioremap %08lx failed\n",
			F286_XDSP_PHYS);
		return -ENXIO;
	}
	for (i = 0; i < ARRAY_SIZE(f286_xdsp_core_tab) && !ret; i++)
		ret = f286_xdsp_boot_core(rad, base, &f286_xdsp_core_tab[i],
					  img);
	iounmap(base);
	kvfree(img);
	return ret;
}

static int f286_load_lte_locked(struct f286_radio *rad)
{
	void __iomem *dram;
	bool dram_ioun;
	u32 head0, head1, entry0, tick;
	unsigned int i;
	int ret;

	/* Stock's NV param setup (items 0x4c/0x22/5, 0x80c10ea8) is skipped:
	 * there is no NV driver yet. This is the LTE RF calibration download
	 * into DSP memory at 0x80678000; see docs/bringup-status.md.
	 */
	dev_info(
		rad->dev,
		"lte: NV param setup (items 0x4c/0x22/5) SKIPPED - no NV driver\n");

	ret = f286_get_mmio_locked(rad, &rad->pmu, F286_PMU_PHYS, F286_PMU_SPAN,
				   "pmu-0x402b");
	if (!ret)
		ret = f286_get_mmio_locked(rad, &rad->aon, F286_AON_PHYS,
					   F286_AON_SPAN, "aon-0x402e");
	if (!ret)
		ret = f286_get_dram_locked(rad, &rad->lte_win,
					   &rad->lte_win_ioun,
					   F286_LTE_WIN_PHYS, F286_WIN_SPAN,
					   "lte-window-0x80400000");
	if (ret)
		return ret;
	if (stage) {
		dev_info(
			rad->dev,
			"lte: stage=1 -> window mapped (lte_win=%p), stopping before firmware load\n",
			rad->lte_win);
		return 0;
	}

	ret = f286_load_firmware(rad, rad->fw_names[0], F286_LTE_DRAM_PHYS,
				 F286_LTE_DRAM_SPAN);
	if (ret)
		return ret;
	dram = f286_map_dram(rad->dev, F286_LTE_DRAM_PHYS, F286_LTE_DRAM_SPAN,
			     &dram_ioun);
	if (!dram)
		return -ENXIO;
	head0 = readl(dram);
	head1 = readl(dram + 4);
	entry0 = readl(dram + 0x400);
	dev_info(rad->dev, "lte: head %08x %08x, entry@0x80448400 %08x\n",
		 head0, head1, entry0);

	/* Stock pre_boot order (model/boot_flow.h LteDspBoot::boot):
	 * start-lo+hold -> 4 PMU clears -> cores poll to 0 -> staging -> ring
	 * -> clock -> boot-addr -> start-ctrl -> release. Every block below
	 * logs its number so dmesg shows the order that actually ran.
	 */
	dev_info(
		rad->dev,
		"lte: stock pre_boot order: [1]start-lo+hold [2]PMU clears [3]cores poll [4]staging [5]ring [6]clock [7]boot-addr [8]start-ctrl [9]release\n");

	/* [1] start-ctrl low + hold reset (bit16). */
	dev_info(rad->dev, "lte: [1] start-ctrl low + hold reset (bit16)\n");
	f286_rmw32(rad, "aon+24c", rad->aon, F286_AON_LTE_START, 0, 1);
	f286_rmw32(rad, "aon+130", rad->aon, F286_AON_DSP_RESET, 0x10000, 0);

	/* [2] Stock's LTE pre-boot 0x80d0fa7c(1) runs these four PMU bank2
	 * CLEAR writes on every cold boot, before staging and boot-addr
	 * (model/boot_hal.h release_power(true)). 0x402b0034 b25:24 are the
	 * core power-request bits (clear = run, set = hold). Before this was
	 * added, the driver wrote 0x402b2044 = 0x30000 (a different sub-field,
	 * 0044 b17:16) and the cores stayed held off (c0[9:5]/[14:10] == 7)
	 * on every hardware run.
	 */
	dev_info(
		rad->dev,
		"lte: [2] stock pre-boot clears: 0044=%08x 0034=%08x 00cc=%08x (pre)\n",
		readl(rad->pmu + F286_PMU_RD_0044),
		readl(rad->pmu + F286_PMU_RD_0034),
		readl(rad->pmu + F286_PMU_RD_00CC));
	writel(0x02000000, rad->pmu + F286_PMU_CLR_2044); /* clr 0044 b25 */
	writel(0x01000000, rad->pmu + F286_PMU_CLR_2044); /* clr 0044 b24 */
	writel(0x00020000, rad->pmu + F286_PMU_CLR_DSP_CFG); /* clr 00cc b17 */
	writel(0x03000000, rad->pmu + F286_PMU_CLR_2034); /* clr 0034 b25:24 */
	dev_info(rad->dev,
		 "lte: [2] post-clear rb: 0044=%08x 0034=%08x 00cc=%08x\n",
		 readl(rad->pmu + F286_PMU_RD_0044),
		 readl(rad->pmu + F286_PMU_RD_0034),
		 readl(rad->pmu + F286_PMU_RD_00CC));

	/* [3] Stock success criterion (HAL 0x811d0d96): XDSP status fields
	 * must read 0: poll c0 until A[9:5]==0 AND B[14:10]==0, ~3 s
	 * (clears 0x811d0df8 BEFORE poll 0x811d0e00), logging every sample.
	 * This replaces the old island-only criterion for the DSP path
	 * (radio_power keeps its own WTLCP criterion).
	 */
	/* Stock HAL 0x811d0d96 polls ONE dsp id per call: id2 -> c0[14:10]
	 * (LDSP). The LTE pre-boot releases only the LDSP request
	 * (PD_WTLCP_LDSP_CFG); TGDSP[9:5] stays held (GSM core) and must NOT
	 * gate the poll. A poll for both cores sat for ~60 s on hardware and
	 * the phone hung the bus inside it.
	 */
	dev_info(
		rad->dev,
		"lte: [3] cores poll: LDSP c0[14:10] -> 0 (0x7c00, stock id2)\n");
	{
		unsigned int tries;
		bool ldsp_on = false;

		for (tries = 0; tries < 200; tries++) {
			u32 c0v = readl(rad->pmu + F286_PMU_POWER_STATUS);

			ldsp_on = ((c0v & 0x7c00) == 0);
			if ((tries % 50) == 0)
				dev_info(
					rad->dev,
					"lte: ldsp poll %u: c0=%08x LDSP[14:10]=%u TGDSP[9:5]=%u\n",
					tries, c0v, (c0v >> 10) & 0x1f,
					(c0v >> 5) & 0x1f);
			if (ldsp_on)
				break;
			/* Sleep with the lock released (the poll must
			 * not pin rad->lock).
			 */
			mutex_unlock(&rad->lock);
			msleep_interruptible(1);
			mutex_lock(&rad->lock);
		}
		dev_info(
			rad->dev,
			"lte: [3] LDSP %s after %u samples (TGDSP is the GSM core: not released on the LTE path)\n",
			ldsp_on ? "ON - power request released" :
				  "STILL HELD after 500 tries",
			tries + 1);
	}

	/* [4] Pre-boot staging (nv9|tick<<16). The stock literal is AP
	 * 0x80447E00, in the second window base 0x80447000 (handshake cells;
	 * sleep-sync 0x80447E14, awake word 0x80447C40), not win+0x7e00
	 * (0x80407E00), which came from attributing the offset to the wrong
	 * base. Value: nv9=1 (nvcal record 2499 @0x03F71C) | tick<<16.
	 */
	ret = f286_get_dram_locked(rad, &rad->hs_win, &rad->hs_win_ioun,
				   F286_HS_WIN_PHYS, F286_HS_WIN_SPAN,
				   "hs-window-0x80447000");
	if (ret)
		return ret;
	tick = (u32)(jiffies / HZ) & 0xffffu;
	writel(1u | (tick << 16), rad->hs_win + F286_HS_NV9);
	dev_info(
		rad->dev,
		"lte: [4] staged nv9|tick -> 0x80447E00 = %08x (nv9=1, tick=%04x; NV items 5/0x22/0x4c SKIPPED, AP-side)\n",
		readl(rad->hs_win + F286_HS_NV9), tick);
	dev_info(
		rad->dev,
		"lte: hs cells baseline: sync@0x80447E14=%08x awake@0x80447C40=%08x\n",
		readl(rad->hs_win + F286_HS_SYNC),
		readl(rad->hs_win + F286_HS_AWAKE));
	/* Second LTE mapping for the split-msg slot region
	 * win+0x33b000..+0x360000 (4 x 0x2800 split slots + 10 x UL-data
	 * slots, stock SPLIT_BODY_OFF = abs 0x8073b000). The main window
	 * map stops at 0x12000 and never covered it. Best-effort: failure
	 * only disables >0x190 payloads (queue_send returns -ENOSPC).
	 */
	if (!rad->split_win) {
		ret = f286_get_dram_locked(
			rad, &rad->split_win, &rad->split_win_ioun,
			F286_LTE_WIN_PHYS + F286_LTE_SPLIT_OFF,
			F286_LTE_SPLIT_SPAN, "lte-split-0x8073b000");
		if (ret)
			dev_warn(
				rad->dev,
				"lte: split-slot map failed %d - payloads > 0x190 unavailable\n",
				ret);
	}
	/* [5] Format the command ring before release: the DSP polls this
	 * ring itself, and a cmd-6 into an unformatted (DRAM
	 * garbage) ring is a no-op. ctl {in=0, out=0}, zeroed 8-entry array,
	 * zeroed body head (queue_send's first alloc lands at +0xe4e0).
	 */
	writel(0, rad->lte_win + F286_Q_CTL);
	for (i = 0; i < 8 * 8; i += 4)
		writel(0, rad->lte_win + F286_Q_ENTRY + i);
	/* Resp queue too (in/out @0xE478, 12 entries
	 * @0xE47C): this window DRAM is not covered by the fw image, so
	 * without this the resp parser would start from garbage indices.
	 */
	for (i = 0; i < 12 * 8; i += 4)
		writel(0, rad->lte_win + F286_Q_RSP_ENTRY + i);
	writel(0, rad->lte_win + F286_Q_RSP_CTL);
	writel(0, rad->lte_win + F286_Q_BODY);
	writel(0, rad->lte_win + F286_Q_BODY + 4);
	dev_info(
		rad->dev,
		"lte: [5] queue ctl initialized (in=0 out=0, readback %08x), entries+body+resp zeroed\n",
		readl(rad->lte_win + F286_Q_CTL));

	/* [6] clock: AON DSP clock-en, then the full XDSP clock-on (light
	 * path, whale enables, TWPLL DSP branch, 0x402D02B4 mux; the
	 * 0x60831044=0x40 tail after the dieclk decision).
	 */
	dev_info(
		rad->dev,
		"lte: [6] clock en + xdsp clock-on (0x60831044<-40 tail, 0x402D02B4 mux=TWPLL_51M2)\n");
	f286_rmw32(rad, "aon+06c", rad->aon, F286_AON_CLK_EN, 0x9620, 0);
	f286_xdsp_clock_on(rad); /* XDSP clock-on (TWPLL_51M2) before release */

	/* [6b] XDSP cores A/B (opt-in; stock boots them long before the LTE
	 * release). A failure is logged and the LTE boot continues.
	 */
	if (xdsp_cores) {
		dev_info(rad->dev,
			 "lte: [6b] XDSP cores A/B boot (xdsp_cores=1)\n");
		f286_step_begin(rad, "xdsp-cores");
		ret = f286_xdsp_boot_cores(rad);
		f286_step_done(rad, "xdsp-cores");
		if (ret)
			dev_err(rad->dev,
				"lte: [6b] XDSP cores failed %d, continuing\n",
				ret);
	}

	/* [7] boot-addr. */
	dev_info(
		rad->dev,
		"lte: [7] boot-addr 0x402e0248 = %08x (mode %u: 0=corrected>>1 0x40224200, 1=raw 0x80448400, 2=corrected+4; entry 0x80448400)\n",
		f286_lte_boot_addr(), boot_addr_mode);
	f286_step_begin(rad, "boot-addr");
	writel(f286_lte_boot_addr(), rad->aon + F286_AON_LTE_BOOT_ADDR);
	dev_info(rad->dev, "lte: boot-addr readback %08x\n",
		 readl(rad->aon + F286_AON_LTE_BOOT_ADDR));
	f286_step_done(rad, "boot-addr");
	udelay(100);
	/* Stock 0x80e2d9e6 runs start_ctrl(true) = [0x402e024c] bit0=1 (HAL
	 * 0x811d0d68) between boot-addr and release. Stock never releases
	 * the DSP with start-ctrl low.
	 */
	dev_info(rad->dev, "lte: [8] start-ctrl high (bit0, HAL 0x811d0d68)\n");
	f286_step_begin(rad, "start-ctrl");
	f286_rmw32(rad, "aon+24c", rad->aon, F286_AON_LTE_START, 1, 0);
	f286_step_done(rad, "start-ctrl");
	/* Pre-release evidence: 16-byte exec vector at the entry + an 8-word
	 * window straddling it + the image head decode. Distinguishes an unclocked window artifact from genuine image
	 * content if the DSP stays mute (compare against the +2 s dump).
	 */
	dev_info(rad->dev,
		 "lte: exec vector @0x80448400: %08x %08x %08x %08x\n",
		 readl(dram + 0x400), readl(dram + 0x404), readl(dram + 0x408),
		 readl(dram + 0x40c));
	dev_info(
		rad->dev,
		"lte: entry window -0x40..+0x40: %08x %08x %08x %08x %08x %08x %08x %08x\n",
		readl(dram + 0x3c0), readl(dram + 0x3c4), readl(dram + 0x3c8),
		readl(dram + 0x3cc), readl(dram + 0x3d0), readl(dram + 0x3d4),
		readl(dram + 0x3d8), readl(dram + 0x3dc));
	dev_info(
		rad->dev,
		"lte: head decode: %08x (0x5a5a5a00 = fill/size marker), %08x (candidate instruction word); boot-addr mode %u (%s), stock codes the reg as entry >> 1\n",
		head0, head1, boot_addr_mode,
		boot_addr_mode == 1 ? "RAW (unshifted)" :
				      (boot_addr_mode == 2 ? "corrected+4" :
							     "corrected >>1"));
	dev_info(rad->dev,
		 "lte: [9] release (bit16 clear, entry 0x80448400)\n");
	f286_step_begin(rad, "release");
	f286_rmw32(rad, "aon+130", rad->aon, F286_AON_DSP_RESET, 0,
		   0x10000); /* RELEASE */
	f286_release_readback(rad, 0x10000, "lte"); /* bit16 stuck check */
	f286_step_done(rad, "release");
	dev_info(rad->dev,
		 "lte: released, entry 0x80448400, queue 0x8040e400\n");
	/* Post-release vector + window re-read after a 2 s settle: if the
	 * words changed, the DSP is fetching (alive) and the silence is an
	 * interface-side problem; compare the queue ctl/body against the
	 * put path in f286_queue_send. The 2 s settle runs with the lock
	 * released.
	 */
	mutex_unlock(&rad->lock);
	msleep_interruptible(2000);
	mutex_lock(&rad->lock);
	{
		u32 v1 = readl(dram + 0x400), v2 = readl(dram + 0x404);
		u32 v3 = readl(dram + 0x408), v4 = readl(dram + 0x40c);
		u32 q = rad->lte_win ? readl(rad->lte_win + F286_Q_CTL) : 0;
		/* Post-release evidence: a change from the pre-release values
		 * (see "XDSP status c0=" and radio_power's power line) is
		 * itself core-activity evidence.
		 */
		u32 c0 = readl(rad->pmu + F286_PMU_POWER_STATUS);
		u32 c4 = readl(rad->pmu + F286_PMU_WTLCP_STATUS);
		u32 d4 = readl(rad->pmu + F286_PMU_WHALE_FSM);

		dev_info(
			rad->dev,
			"lte: exec vector AFTER release (+2 s): %08x %08x %08x %08x [%s]\n",
			v1, v2, v3, v4,
			(v1 != entry0) ? "CHANGED - DSP FETCHING" :
					 "unchanged - see head decode");
		dev_info(
			rad->dev,
			"lte: post-release +2s c0=%08x (XDSP-A[9:5]=%u XDSP-B[14:10]=%u) c4=%08x (WTLCP[4:0]=%u) d4=%08x (state=%u)\n",
			c0, (c0 >> 5) & 0x1f, (c0 >> 10) & 0x1f, c4, c4 & 0x1f,
			d4, (d4 >> 4) & 0xf);
		dev_info(
			rad->dev,
			"lte: post-release queue ctl=%08x in=%04x out=%04x (fixed-8 put-path: entry {6,8,end_off}, body 8 raw bytes at +0xe4e0; headered msgs per hdr_layout)\n",
			q, q & 0xffffu, q >> 16);
	}
	f286_unmap_dram(&dram, dram_ioun);

	/* The stock cold boot sends no command to the DSP; the handshake is
	 * DSP-initiated (unprompted resp type 5 version, drained by the pump
	 * kthread). cmdId 6 is opt-in via send_cmd6. Fixed-8 shape: no
	 * message header, entry.len = 8, entry.cmdId = 6.
	 */
	if (send_cmd6) {
		f286_step_begin(rad, "cmd6");
		f286_queue_send(rad, rad->lte_win, 6, f286_lte_cmd6_template,
				sizeof(f286_lte_cmd6_template), true);
		f286_step_done(rad, "cmd6");
	} else {
		dev_info(
			rad->dev,
			"lte: post-release cmd6 skipped (send_cmd6=0; the stock cold boot sends nothing, wait for resp type 5)\n");
	}
	return 0;
}

/* ---- L1 resp pump (dedicated kthread polling at 5 ms, like stock) ----
 * The 12-slot resp ring overflows in a DL burst if it is only drained every
 * 5 s. Pump semantics follow the stock pump 0x80c109a0 (model/l1ring.h):
 * decode type 5 (VERSION) / 7 (ASSERT), log
 * every other entry, out = (out+1) % 12 for EVERY consumed entry, bound
 * addr+len < 0x100c. `out` is AP-owned (DSP writes only `in`).
 */

static void f286_resp_pump_win(struct f286_radio *rad, void __iomem *win,
			       const char *name)
{
	u32 rctl = readl(win + F286_Q_RSP_CTL);
	u16 rsp_in = rctl & 0xffffu;
	u16 rsp_out = rctl >> 16;
	unsigned int n;

	if (rsp_in >= F286_Q_RSP_MOD) {
		dev_warn(rad->dev,
			 "%s resp: in index %u out of range, pump skipped\n",
			 name, rsp_in);
		return;
	}
	if (rsp_out >= F286_Q_RSP_MOD)
		rsp_out = 0; /* sanitize garbage: AP owns this index */
	for (n = 0; rsp_out != rsp_in && n < F286_Q_RSP_MOD; n++) {
		u32 e0 = readl(win + F286_Q_RSP_ENTRY + rsp_out * 8);
		u32 ea = readl(win + F286_Q_RSP_ENTRY + rsp_out * 8 + 4);
		u16 rtype = e0 & 0xffffu;
		u16 rlen = e0 >> 16;
		u32 raddr = ea;

		dev_info(
			rad->dev,
			"%s RESP[%u]: type %u len %u addr %06x (raw %08x %08x)\n",
			name, rsp_out, rtype, rlen, raddr, e0, ea);
		if (raddr < 0x100cu && rlen <= 0x100cu &&
		    raddr + rlen < 0x100cu) {
			/* Log only the bytes the response actually
			 * carries.
			 */
			unsigned int rb = min_t(unsigned int, rlen, 8);
			u32 p0 = 0;
			u32 p1 = 0;

			if (rb > 0)
				p0 = readl(win + F286_Q_RSP_BODY + raddr);
			if (rb > 4)
				p1 = readl(win + F286_Q_RSP_BODY + raddr + 4);

			dev_info(rad->dev, "%s resp payload: %08x %08x\n", name,
				 p0, p1);
			if (rtype == 5) {
				WRITE_ONCE(rad->l1_type5, true);
				dev_info(
					rad->dev,
					"LTEDSP_Ver handshake SUCCESS (unprompted type 5): branch=%u svn=%u chip_id=%u chip_ver=%u\n",
					p0 & 0xffffu, p0 >> 16, p1 & 0xffffu,
					p1 >> 16);
			} else if (rtype == 7) {
				dev_err(rad->dev, "%s DSP ASSERT: %08x %08x\n",
					name, p0, p1);
			}
		}
		rsp_out = (rsp_out + 1) % F286_Q_RSP_MOD;
		writew(rsp_out, win + F286_Q_RSP_CTL + 2);
		rad->poll_seen_change = true;
	}
}

static int f286_pump_thread(void *data)
{
	struct f286_radio *rad = data;

	dev_info(rad->dev, "l1 pump thread enter (5 ms resp poll)\n");
	while (!kthread_should_stop()) {
		/* Windows are only unmapped from remove AFTER kthread_stop,
		 * so these lock-free reads are safe.
		 */
		if (rad->lte_win)
			f286_resp_pump_win(rad, rad->lte_win, "lte");
		if (rad->gsm_win)
			f286_resp_pump_win(rad, rad->gsm_win, "gsm");
		msleep_interruptible(5);
	}
	rad->pumper = NULL;
	return 0;
}

/* ---- load thread: sequence under the lock, 90 s poll lock-free ---- */

static int f286_load_thread(void *data)
{
	struct f286_radio *rad = data;
	bool lte = rad->load_lte_req;
	const char *name = lte ? "lte" : "gsm";
	void __iomem *win;
	/* On the heap: as 2 x 3 KB stack locals they made a 6256-byte frame. */
	u32 *snap, *cur;
	u16 cmd_out;
	unsigned int i, it;
	int ret;

	snap = kvmalloc_array(F286_Q_WATCH / 4, sizeof(u32),
			      GFP_KERNEL | __GFP_NORETRY);
	cur = kvmalloc_array(F286_Q_WATCH / 4, sizeof(u32),
			     GFP_KERNEL | __GFP_NORETRY);
	if (!snap || !cur) {
		dev_err(rad->dev, "load thread (%s): poll buffers OOM\n", name);
		kvfree(snap);
		kvfree(cur);
		rad->loader = NULL;
		return -ENOMEM;
	}
	memset(snap, 0, F286_Q_WATCH);
	memset(cur, 0, F286_Q_WATCH);

	dev_info(rad->dev, "load thread (%s) enter\n", name);

	/* Hardware sequence under the lock. The sequence contains bounded
	 * sleep polls, but every multi-ms sleep runs with the lock released
	 * (unlock/msleep/lock pairs),
	 * so the held stretches contain bus accesses only. Includes the
	 * optional (send_cmd6, default off) LTE cmdId 6 send right after
	 * release.
	 */
	mutex_lock(&rad->lock);
	ret = lte ? f286_load_lte_locked(rad) : f286_load_gsm_locked(rad);
	if (ret == 0)
		rad->state = lte ? F286_ST_LTE_RUNNING : F286_ST_GSM_RUNNING;
	mutex_unlock(&rad->lock);
	if (ret) {
		dev_err(rad->dev, "load thread (%s): sequence failed %d\n",
			name, ret);
		kvfree(snap);
		kvfree(cur);
		rad->loader = NULL;
		return ret;
	}

	/* Spawn the persistent resp-pump kthread (5 ms) before the poll
	 * window: the 12-slot resp ring overflows in a DL burst if it is only
	 * drained every 5 s, and a pump inside this loop would die with the
	 * load thread. The pump keeps running
	 * after this thread exits and is stopped only in remove.
	 */
	WRITE_ONCE(rad->l1_type5, false);
	mutex_lock(&rad->lock);
	if (!rad->pumper) {
		struct task_struct *t =
			kthread_run(f286_pump_thread, rad, "f286-l1pump");

		if (IS_ERR(t)) {
			dev_err(rad->dev,
				"l1 pump kthread spawn FAILED %pe - the resp ring will NOT drain\n",
				t);
		} else {
			rad->pumper = t;
			dev_info(rad->dev, "l1 pump kthread spawned\n");
		}
	}
	mutex_unlock(&rad->lock);

	/* 90 s poll, LOCK-FREE, interruptible, 5 s granularity: the mutex is
	 * NOT held while sleeping (status and other ops stay live). The DSP
	 * polls the ring itself; there is no doorbell. Resp draining is the
	 * pump kthread's job; this loop keeps the byte-diff watch + the final
	 * POLL RESULT verdict.
	 */
	win = lte ? rad->lte_win : rad->gsm_win;
	memcpy_fromio(snap, win + F286_Q_CTL, F286_Q_WATCH);
	cmd_out = snap[0] >> 16;
	dev_info(
		rad->dev,
		"%s poll start: ctl=%08x in=%04x out=%04x resp ctl=%08x (pump kthread drains)\n",
		name, snap[0], snap[0] & 0xffffu, snap[0] >> 16,
		readl(win + F286_Q_RSP_CTL));

	for (it = 1; it <= 18 && !kthread_should_stop(); it++) {
		msleep_interruptible(5000);
		if (kthread_should_stop())
			break;

		memcpy_fromio(cur, win + F286_Q_CTL, F286_Q_WATCH);
		for (i = 0; i < F286_Q_WATCH / 4; i++) {
			if (cur[i] != snap[i]) {
				dev_info(
					rad->dev,
					"%s CHANGE +%04x: %08x -> %08x  <<< DSP ACTIVITY\n",
					name,
					(unsigned int)(F286_Q_CTL + i * 4),
					snap[i], cur[i]);
				snap[i] = cur[i];
				rad->poll_seen_change = true;
			}
		}
		/* Act on ring movement instead of only byte-diffing it. Main ctl `out` (high half) advancing
		 * = the DSP consumed our cmd entries (AP owns `in`;
		 * log-only).
		 */
		{
			u32 ctl = readl(win + F286_Q_CTL);

			if ((ctl >> 16) != cmd_out) {
				dev_info(
					rad->dev,
					"%s cmds consumed: out %u->%u (in %u)\n",
					name, cmd_out, ctl >> 16,
					ctl & 0xffffu);
				cmd_out = ctl >> 16;
				rad->poll_seen_change = true;
			}
		}
		/* Resp draining lives in the f286-l1pump kthread (5 ms
		 * cadence: decode 5/7, log others, out=(out+1)%12, bound
		 * addr+len < 0x100c).
		 */
		if ((it % 3) == 0)
			dev_info(
				rad->dev,
				"%s poll progress %us: ctl=%08x in=%04x out=%04x changes=%d\n",
				name, it * 5, cur[0], cur[0] & 0xffffu,
				cur[0] >> 16, rad->poll_seen_change);
		/* Watch words beyond the 0xc00 diff span: stock alive-check
		 * dump word (+0x10bf8, 0x33 = dump), sleep/resume
		 * sync (+0xe14, DSP writes 0x22222222), window head
		 * words (+0x400), plus the REAL handshake cells in the
		 * 0x80447000 window (nv9 staging 0x80447E00, awake
		 * 0x80447C40, sleep-sync 0x80447E14) when mapped. ANY
		 * change means the DSP is alive even if it never
		 * touches the ring. These run every 5 s iteration, not only
		 * on the progress lines.
		 */
		if (rad->hs_win)
			dev_info(
				rad->dev,
				"%s watch hs: 47e00=%08x 47c40=%08x 47e14=%08x\n",
				name, readl(rad->hs_win + F286_HS_NV9),
				readl(rad->hs_win + F286_HS_AWAKE),
				readl(rad->hs_win + F286_HS_SYNC));
		dev_info(
			rad->dev,
			"%s watch: +10bf8=%08x +e14=%08x +400=%08x %08x %08x %08x\n",
			name, readl(win + F286_WIN_ALIVE_DUMP),
			readl(win + F286_WIN_SLEEP_SYNC), readl(win + 0x400),
			readl(win + 0x404), readl(win + 0x408),
			readl(win + 0x40c));
	}
	/* Verdict: unprompted resp type 5 received = handshake success
	 * (LTEDSP_Ver fields logged by the pump kthread at receive time); no
	 * type 5 in 90 s = silent. Things to try next when silent:
	 * boot_addr_mode=1 + reload, then =2, then dieclk=1.
	 */
	if (READ_ONCE(rad->l1_type5))
		dev_info(
			rad->dev,
			"%s POLL RESULT: HANDSHAKE SUCCESS (resp type 5 received; LTEDSP_Ver above) ctl=%08x in=%04x out=%04x changes=%d\n",
			name, cur[0], cur[0] & 0xffffu, cur[0] >> 16,
			rad->poll_seen_change);
	else
		dev_err(rad->dev,
			"%s POLL RESULT: SILENT (no type 5 in 90 s) ctl=%08x in=%04x out=%04x changes=%d - DSP never handshook; next: boot_addr_mode=1 + reload, then =2, then dieclk=1\n",
			name, cur[0], cur[0] & 0xffffu, cur[0] >> 16,
			rad->poll_seen_change);
	/* The pump kthread stays running after this thread exits;
	 * f286-l1pump drains both resp rings until the driver is removed.
	 */
	kvfree(snap);
	kvfree(cur);
	rad->loader = NULL;
	return 0;
}

static int f286_load_spawn_locked(struct f286_radio *rad, bool lte)
{
	/* A load may not interleave with an in-flight radio_power (and vice
	 * versa below): both mutate the same PMU/AON state.
	 */
	if (rad->loader || rad->powerer)
		return -EBUSY;
	rad->load_lte_req = lte;
	rad->loader = kthread_run(f286_load_thread, rad, "f286-%s-load",
				  lte ? "lte" : "gsm");
	if (IS_ERR(rad->loader)) {
		int err = PTR_ERR(rad->loader);

		rad->loader = NULL;
		return err;
	}
	return 0;
}

/* ---- sysfs ---- */

/* radio_power runs in its own kthread so the sysfs store returns
 * immediately: a kernfs store must never sit in the ~30 s WTLCP poll
 * holding rad->lock (and its kernfs active ref) across msleep-class
 * waits. The sequence still runs under rad->lock, but every sleep inside
 * f286_radio_power_locked releases it.
 */
static int f286_power_thread(void *data)
{
	struct f286_radio *rad = data;
	int ret;

	mutex_lock(&rad->lock);
	ret = f286_radio_power_locked(rad);
	mutex_unlock(&rad->lock);
	dev_info(rad->dev, "radio_power: power thread done (%d, state=%s)\n",
		 ret, f286_state_names[rad->state]);
	rad->powerer = NULL;
	return ret;
}

static ssize_t radio_power_store(struct device *dev,
				 struct device_attribute *attr,
				 const char *buffer, size_t count)
{
	struct f286_radio *rad = dev_get_drvdata(dev);
	unsigned int value;
	int ret;

	ret = kstrtouint(buffer, 0, &value);
	if (ret)
		return ret;
	if (value != 1)
		return -EINVAL;

	/* Spawn, don't wait: the store returns immediately; the
	 * outcome lands in dmesg + rad->state.
	 */
	mutex_lock(&rad->lock);
	if (rad->powerer || rad->loader || rad->state == F286_ST_POWERING) {
		mutex_unlock(&rad->lock);
		return -EBUSY;
	}
	rad->powerer = kthread_run(f286_power_thread, rad, "f286-power");
	if (IS_ERR(rad->powerer)) {
		ret = PTR_ERR(rad->powerer);
		rad->powerer = NULL;
		mutex_unlock(&rad->lock);
		return ret;
	}
	mutex_unlock(&rad->lock);
	dev_info(
		rad->dev,
		"radio_power: power thread started (store returns immediately)\n");
	return count;
}
static int f286_axidump(struct f286_radio *rad, char *buf, size_t size);

static ssize_t axidump_show(struct device *dev, struct device_attribute *attr,
			    char *buf)
{
	struct f286_radio *rad = dev_get_drvdata(dev);

	return f286_axidump(rad, buf, 4096);
}
static DEVICE_ATTR_RO(axidump);

/* gatetest: hypothesis that AUTO_GATE_CTRL0 (0x402e0140) bits [7:6]
 * (cgm_wtl_bridge_en) gate the AP<->whale bridge. Writing 0xC0 to the SET bank
 * (0x402e1140) may make the whale-APB (0x6083) answer bus transactions that
 * currently read as 0 / drop writes. One-shot experiment.
 */
static ssize_t gatetest_store(struct device *dev, struct device_attribute *attr,
			      const char *buf, size_t count)
{
	struct f286_radio *rad = dev_get_drvdata(dev);

	if (!rad->aon || !rad->whale_apb)
		return -EIO;
	dev_info(rad->dev, "gatetest: pre 0x60831034=%08x 1044=%08x\n",
		 readl(rad->whale_apb + 0x1034),
		 readl(rad->whale_apb + 0x1044));
	dev_info(rad->dev, "gatetest: pre 0x402e0140=%08x\n",
		 readl(rad->aon + 0x140));
	writew(0xc0, rad->aon + 0x1140); /* AUTO_GATE_CTRL0 SET b7:6 */
	dev_info(rad->dev, "gatetest: 0x402e1140 <- c0, 0x402e0140 now %08x\n",
		 readl(rad->aon + 0x140));
	writel(0x40000, rad->whale_apb + 0x1034);
	dev_info(rad->dev, "gatetest: 0x60831034 <- 40000, rb %08x\n",
		 readl(rad->whale_apb + 0x1034));
	writel(0x2000, rad->whale_apb + 0x1060);
	dev_info(rad->dev, "gatetest: 0x60831060 <- 2000, rb %08x\n",
		 readl(rad->whale_apb + 0x1060));
	writel(0x80, rad->whale_apb + 0x1044);
	dev_info(rad->dev, "gatetest: 0x60831044 <- 80, rb %08x\n",
		 readl(rad->whale_apb + 0x1044));
	return count;
}
static DEVICE_ATTR_WO(gatetest);

static DEVICE_ATTR_WO(radio_power);

static ssize_t load_gsm_store(struct device *dev, struct device_attribute *attr,
			      const char *buffer, size_t count)
{
	struct f286_radio *rad = dev_get_drvdata(dev);
	unsigned int value;
	int ret;

	ret = kstrtouint(buffer, 0, &value);
	if (ret)
		return ret;
	if (value != 1)
		return -EINVAL;

	mutex_lock(&rad->lock);
	ret = f286_load_spawn_locked(rad, false);
	mutex_unlock(&rad->lock);
	if (ret)
		return ret;
	dev_info(rad->dev, "load_gsm: load thread started\n");
	return count;
}
static DEVICE_ATTR_WO(load_gsm);

static ssize_t load_lte_store(struct device *dev, struct device_attribute *attr,
			      const char *buffer, size_t count)
{
	struct f286_radio *rad = dev_get_drvdata(dev);
	unsigned int value;
	int ret;

	ret = kstrtouint(buffer, 0, &value);
	if (ret)
		return ret;
	if (value != 1)
		return -EINVAL;

	mutex_lock(&rad->lock);
	ret = f286_load_spawn_locked(rad, true);
	mutex_unlock(&rad->lock);
	if (ret)
		return ret;
	dev_info(rad->dev, "load_lte: load thread started\n");
	return count;
}
static DEVICE_ATTR_WO(load_lte);

static ssize_t status_show(struct device *dev, struct device_attribute *attr,
			   char *buffer)
{
	struct f286_radio *rad = dev_get_drvdata(dev);
	u32 c0 = 0, c4 = 0, d4 = 0, gq = 0, lq = 0;
	ssize_t bytes = 0;

	/* Lock-free diagnostic snapshot: status must never block on a mutex
	 * held by an in-flight load/poll (racy reads are fine here).
	 */
	if (rad->pmu) {
		c0 = readl(rad->pmu + F286_PMU_POWER_STATUS);
		c4 = readl(rad->pmu + F286_PMU_WTLCP_STATUS);
		d4 = readl(rad->pmu + F286_PMU_WHALE_FSM);
	}
	if (rad->gsm_win)
		gq = readl(rad->gsm_win + F286_Q_CTL);
	if (rad->lte_win)
		lq = readl(rad->lte_win + F286_Q_CTL);
	bytes += sysfs_emit_at(buffer, bytes, "state=%s\n",
			       f286_state_names[rad->state]);
	if (rad->pmu)
		bytes += sysfs_emit_at(
			buffer, bytes,
			"c0=%08x whu3g[4:0]=%u xdspb=%u xdspa=%u cm4[19:15]=%u\n"
			"c4=%08x wtlcp[4:0]=%u\n"
			"d4=%08x whale_state[7:4]=%u\n",
			c0, c0 & 0x1f, (c0 >> 5) & 0x1f, (c0 >> 10) & 0x1f,
			(c0 >> 15) & 0x1f, c4, c4 & 0x1f, d4, (d4 >> 4) & 0xf);
	else
		bytes += sysfs_emit_at(buffer, bytes, "pmu=unmapped\n");
	if (rad->gsm_win)
		bytes += sysfs_emit_at(buffer, bytes,
				       "gsm_queue ctl=%08x in=%04x out=%04x\n",
				       gq, gq & 0xffffu, gq >> 16);
	else
		bytes += sysfs_emit_at(buffer, bytes, "gsm_queue=unmapped\n");
	if (rad->lte_win)
		bytes += sysfs_emit_at(buffer, bytes,
				       "lte_queue ctl=%08x in=%04x out=%04x\n",
				       lq, lq & 0xffffu, lq >> 16);
	else
		bytes += sysfs_emit_at(buffer, bytes, "lte_queue=unmapped\n");
	bytes += sysfs_emit_at(buffer, bytes, "loader=%s\n",
			       rad->loader ? (rad->load_lte_req ?
						      "lte-load-running" :
						      "gsm-load-running") :
					     "idle");
	bytes += sysfs_emit_at(buffer, bytes, "powerer=%s\n",
			       rad->powerer ? "powering" : "idle");
	return bytes;
}
static DEVICE_ATTR_RO(status);

static ssize_t queue_read(struct file *file, struct kobject *kobj,
			  const struct bin_attribute *bin, char *buffer,
			  loff_t pos, size_t count)
{
	struct f286_radio *rad = dev_get_drvdata(kobj_to_dev(kobj));
	size_t copied = 0;

	if (pos < 0)
		return -EINVAL;
	if (pos >= (loff_t)(2 * F286_Q_USER_SPAN))
		return 0;
	if ((pos < F286_Q_USER_SPAN && !rad->gsm_win) ||
	    (pos >= F286_Q_USER_SPAN && !rad->lte_win))
		return -EIO;
	if (count > (2 * F286_Q_USER_SPAN) - pos)
		count = (2 * F286_Q_USER_SPAN) - pos;

	while (copied < count) {
		void __iomem *win;
		u32 off = pos + copied;
		u32 in_win = off % F286_Q_USER_SPAN;
		size_t chunk;

		if (off < F286_Q_USER_SPAN) {
			win = rad->gsm_win;
		} else {
			win = rad->lte_win;
			in_win = off - F286_Q_USER_SPAN;
		}
		if (!win)
			break; /* second window unmapped: return the part read */
		chunk = min_t(size_t, count - copied,
			      F286_Q_USER_SPAN - in_win);
		memcpy_fromio(buffer + copied, win + F286_Q_CTL + in_win,
			      chunk);
		copied += chunk;
	}
	return copied;
}
static BIN_ATTR_RO(queue, 2 * F286_Q_USER_SPAN);

/* Full AP-window dumps: byte N of window_<gsm|lte> = physical
 * (0x800e0000 | 0x80400000) + N, N = 0..0x12000-1. Reads the WHOLE mapped
 * window: ctl/entries/body (0xe400..), sleep-sync (+0xe14), resp queue
 * (+0xe478..), alive-check dump word (+0x10bf8). (The nv9 staging cell is
 * NOT here: stock stages it at AP 0x80447E00, in the separately mapped
 * handshake window 0x80447000.) Any DSP-side write becomes visible, not
 * just the 0xe400..0x10000 slice the queue attr covers. Offsets are listed
 * in docs/radio-driver.md.
 */
static ssize_t f286_window_read(struct f286_radio *rad, void __iomem *win,
				char *buffer, loff_t pos, size_t count)
{
	if (!win)
		return -EIO; /* window not mapped yet (run init / load) */
	if (pos < 0)
		return -EINVAL;
	if ((size_t)pos >= F286_WIN_SPAN)
		return 0;
	if (count > F286_WIN_SPAN - (size_t)pos)
		count = F286_WIN_SPAN - (size_t)pos;
	memcpy_fromio(buffer, win + pos, count);
	return count;
}

static ssize_t window_gsm_read(struct file *file, struct kobject *kobj,
			       const struct bin_attribute *bin, char *buffer,
			       loff_t pos, size_t count)
{
	struct f286_radio *rad = dev_get_drvdata(kobj_to_dev(kobj));

	return f286_window_read(rad, rad->gsm_win, buffer, pos, count);
}
static BIN_ATTR_RO(window_gsm, F286_WIN_SPAN);

static ssize_t window_lte_read(struct file *file, struct kobject *kobj,
			       const struct bin_attribute *bin, char *buffer,
			       loff_t pos, size_t count)
{
	struct f286_radio *rad = dev_get_drvdata(kobj_to_dev(kobj));

	return f286_window_read(rad, rad->lte_win, buffer, pos, count);
}
static BIN_ATTR_RO(window_lte, F286_WIN_SPAN);

static ssize_t init_store(struct device *dev, struct device_attribute *attr,
			  const char *buffer, size_t count)
{
	struct f286_radio *rad = dev_get_drvdata(dev);
	unsigned int value;
	int ret;

	ret = kstrtouint(buffer, 0, &value);
	if (ret)
		return ret;
	if (value != 1)
		return -EINVAL;

	mutex_lock(&rad->lock);
	if (rad->init_done) {
		mutex_unlock(&rad->lock);
		return -EBUSY;
	}
	ret = f286_init_locked(rad);
	mutex_unlock(&rad->lock);
	return ret ? ret : count;
}
static DEVICE_ATTR_WO(init);

/* ---- deferred init (runs with console + SSH up) ---- */

static int f286_init_locked(struct f286_radio *rad)
{
	int ret;

	dev_info(rad->dev, "init sequence enter (autoint=%d no_autoint=%d)\n",
		 autoint, no_autoint);

	ret = f286_get_mmio_locked(rad, &rad->key, F286_KEY_PHYS, F286_KEY_SPAN,
				   "key-0x4031");
	dev_info(rad->dev, "init: key-0x4031 -> %d\n", ret);
	if (!ret) {
		dev_info(rad->dev, "init: ioremap whale-0x6083...\n");
		ret = f286_get_mmio_locked(rad, &rad->whale_apb,
					   F286_WHALE_APB_PHYS,
					   F286_WHALE_APB_SPAN, "whale-0x6083");
		dev_info(rad->dev, "init: whale-0x6083 -> %d\n", ret);
	}
	if (!ret) {
		dev_info(rad->dev, "init: ioremap blk-0x404d...\n");
		ret = f286_get_mmio_locked(rad, &rad->blk404d,
					   F286_BLK404D_PHYS, F286_BLK404D_SPAN,
					   "blk-0x404d");
		dev_info(rad->dev, "init: blk-0x404d -> %d\n", ret);
	}
	if (!ret) {
		dev_info(rad->dev,
			 "init: masking unclaimed baseband IRQs via GICD...\n");
		if (f286_get_mmio_locked(rad, &rad->gicd, F286_GICD_PHYS,
					 F286_GICD_SPAN, "gicd") == 0) {
			/* Same SPI-numbering doubt as the radio_power mask
			 * (hwirq 68/69 look like the WCN mailbox's). Save the
			 * originals (first mask wins) for the remove-time
			 * restore.
			 */
			u32 before = readl(rad->gicd + 0x188);

			if (!rad->gicd_masked) {
				rad->gicd_icenabler2 = before;
				rad->gicd_icenabler3 = readl(rad->gicd + 0x18c);
				rad->gicd_masked = true;
			}
			writel(0xe0000030, rad->gicd + 0x188);
			writel(0xe0000000, rad->gicd + 0x18c);
			dev_info(
				rad->dev,
				"init: GICD_ICENABLER2 %08x->%08x, ICENABLER3 %08x->e0000000 (originals saved for remove restore)\n",
				before, readl(rad->gicd + 0x188),
				rad->gicd_icenabler3);
		} else {
			dev_warn(
				rad->dev,
				"init: GICD unmappable, IRQ mask SKIPPED (unclaimed DSP IRQs may storm)\n");
		}
	}
	if (!ret) {
		dev_info(rad->dev, "init: map gsm-window-0x800e0000...\n");
		ret = f286_get_dram_locked(rad, &rad->gsm_win,
					   &rad->gsm_win_ioun,
					   F286_GSM_WIN_PHYS, F286_WIN_SPAN,
					   "gsm-window-0x800e0000");
		dev_info(rad->dev, "init: gsm-window -> %d\n", ret);
	}
	if (!ret) {
		dev_info(rad->dev, "init: map lte-window-0x80400000...\n");
		ret = f286_get_dram_locked(rad, &rad->lte_win,
					   &rad->lte_win_ioun,
					   F286_LTE_WIN_PHYS, F286_WIN_SPAN,
					   "lte-window-0x80400000");
		dev_info(rad->dev, "init: lte-window -> %d\n", ret);
	}

	rad->init_done = true;
	dev_info(rad->dev, "init: sequence complete\n");
	return ret;
}

static void f286_init_work(struct work_struct *work)
{
	struct f286_radio *rad =
		container_of(work, struct f286_radio, init_work.work);
	int ret, wret;

	mutex_lock(&rad->lock);
	if (!autoint || no_autoint)
		dev_info(
			rad->dev,
			"init work: autoint=0, skipping the radio sequence (maps happen via the init attribute)\n");
	else
		f286_init_locked(rad);
	mutex_unlock(&rad->lock);

	/* One creation call per attr with a log line immediately before
	 * each, so a hang names the attr with console +
	 * USB network up.
	 */
	dev_info(rad->dev, "attrs: creating radio_power...\n");
	ret = sysfs_create_file(&rad->dev->kobj, &dev_attr_radio_power.attr);
	dev_info(rad->dev, "attrs: radio_power -> %d\n", ret);
	dev_info(rad->dev, "attrs: creating status...\n");
	ret = sysfs_create_file(&rad->dev->kobj, &dev_attr_status.attr);
	dev_info(rad->dev, "attrs: status -> %d\n", ret);
	dev_info(rad->dev, "attrs: creating load_lte...\n");
	ret = sysfs_create_file(&rad->dev->kobj, &dev_attr_load_lte.attr);
	dev_info(rad->dev, "attrs: load_lte -> %d\n", ret);
	dev_info(rad->dev, "attrs: creating load_gsm...\n");
	ret = sysfs_create_file(&rad->dev->kobj, &dev_attr_load_gsm.attr);
	dev_info(rad->dev, "attrs: load_gsm -> %d\n", ret);
	dev_info(rad->dev, "attrs: creating axidump...\n");
	ret = sysfs_create_file(&rad->dev->kobj, &dev_attr_axidump.attr);
	dev_info(rad->dev, "attrs: axidump -> %d\n", ret);
	dev_info(rad->dev, "attrs: creating gatetest...\n");
	ret = sysfs_create_file(&rad->dev->kobj, &dev_attr_gatetest.attr);
	dev_info(rad->dev, "attrs: gatetest -> %d\n", ret);
	dev_info(rad->dev, "attrs: creating queue (bin_attr)...\n");
	ret = sysfs_create_bin_file(&rad->dev->kobj, &bin_attr_queue);
	dev_info(rad->dev, "attrs: queue -> %d\n", ret);
	dev_info(rad->dev, "attrs: creating window_gsm (bin_attr)...\n");
	wret = sysfs_create_bin_file(&rad->dev->kobj, &bin_attr_window_gsm);
	dev_info(rad->dev, "attrs: window_gsm -> %d\n", wret);
	dev_info(rad->dev, "attrs: creating window_lte (bin_attr)...\n");
	if (wret == 0)
		wret = sysfs_create_bin_file(&rad->dev->kobj,
					     &bin_attr_window_lte);
	dev_info(rad->dev, "attrs: window_lte -> %d\n", wret);
	if (ret == 0)
		ret = wret;
	if (ret == 0)
		rad->sysfs_ready = true;
	dev_info(rad->dev, "attrs: all attrs %s\n",
		 ret == 0 ? "OK" : "created with errors");
}

/* ---- probe / remove ---- */

static int f286_radio_probe(struct platform_device *pdev)
{
	struct f286_radio *rad;
	const char *names[2] = { "ums9117/f286-lte-dsp.bin",
				 "ums9117/f286-gsm-dsp.bin" };
	int ret;

	/* Minimal probe: allocate, read DT, create the `init` attr (safe on
	 * hardware), schedule the 60 s deferred work. No MMIO maps, no GICD,
	 * no window maps, no other sysfs: console and the USB
	 * network are always up before any of that runs.
	 */
	rad = devm_kzalloc(&pdev->dev, sizeof(*rad), GFP_KERNEL);
	if (!rad)
		return -ENOMEM;
	rad->dev = &pdev->dev;
	mutex_init(&rad->lock);
	platform_set_drvdata(pdev, rad);

	if (device_property_read_string_array(&pdev->dev, "firmware-name",
					      names, 2) != 2)
		dev_info(
			&pdev->dev,
			"firmware-name DT property missing/incomplete, using defaults\n");
	rad->fw_names[0] = names[0];
	rad->fw_names[1] = names[1];

	INIT_DELAYED_WORK(&rad->init_work, f286_init_work);
	ret = device_create_file(&pdev->dev, &dev_attr_init);
	if (ret)
		return ret;
	dev_info(
		&pdev->dev,
		"probe enter: only the init attr exists; the other attrs appear one-by-one from the deferred work at +60 s (autoint=%d no_autoint=%d stage=%d)\n",
		autoint, no_autoint, stage);
	schedule_delayed_work(&rad->init_work, 60 * HZ);
	return 0;
}

static void f286_radio_remove(struct platform_device *pdev)
{
	struct f286_radio *rad = platform_get_drvdata(pdev);

	cancel_delayed_work_sync(&rad->init_work);
	/* Stop the worker threads without holding rad->lock: both
	 * re-acquire it after their lock-free sleep windows, so a stop
	 * under the lock could deadlock.
	 */
	if (rad->loader) {
		kthread_stop(rad->loader);
		rad->loader = NULL;
	}
	if (rad->powerer) {
		kthread_stop(rad->powerer);
		rad->powerer = NULL;
	}
	/* The pump kthread must die before the DSP windows are unmapped
	 * (it lock-freely reads rad->lte_win / rad->gsm_win every 5 ms).
	 */
	if (rad->pumper) {
		kthread_stop(rad->pumper);
		rad->pumper = NULL;
	}
	sysfs_remove_file(&pdev->dev.kobj, &dev_attr_radio_power.attr);
	sysfs_remove_file(&pdev->dev.kobj, &dev_attr_status.attr);
	sysfs_remove_file(&pdev->dev.kobj, &dev_attr_load_lte.attr);
	sysfs_remove_file(&pdev->dev.kobj, &dev_attr_load_gsm.attr);
	sysfs_remove_file(&pdev->dev.kobj, &dev_attr_axidump.attr);
	sysfs_remove_file(&pdev->dev.kobj, &dev_attr_gatetest.attr);
	sysfs_remove_bin_file(&pdev->dev.kobj, &bin_attr_queue);
	sysfs_remove_bin_file(&pdev->dev.kobj, &bin_attr_window_gsm);
	sysfs_remove_bin_file(&pdev->dev.kobj, &bin_attr_window_lte);
	device_remove_file(&pdev->dev, &dev_attr_init);
	f286_unmap_dram(&rad->gsm_win, rad->gsm_win_ioun);
	f286_unmap_dram(&rad->lte_win, rad->lte_win_ioun);
	f286_unmap_dram(&rad->hs_win, rad->hs_win_ioun);
	f286_unmap_dram(&rad->split_win, rad->split_win_ioun);
	/* Restore the original GICD ICENABLERs before the unmap so
	 * disabling the unclaimed IRQs is reversible at driver release (the
	 * masked set plausibly includes hwirq 68/69 = the WCN/BT mailbox,
	 * which matters for any future WCN work).
	 */
	if (rad->gicd_masked && rad->gicd) {
		writel(rad->gicd_icenabler2, rad->gicd + 0x188);
		writel(rad->gicd_icenabler3, rad->gicd + 0x18c);
		dev_info(
			rad->dev,
			"remove: GICD mask restored: ICENABLER2 <- %08x (rb %08x), ICENABLER3 <- %08x (rb %08x)\n",
			rad->gicd_icenabler2, readl(rad->gicd + 0x188),
			rad->gicd_icenabler3, readl(rad->gicd + 0x18c));
	}
	f286_unmap_mmio(&rad->gicd);
	f286_unmap_mmio(&rad->blk404d);
	/* The dieclk kthread (if any) dereferences only its
	 * own private map copies (f286_dieclk_thread), never these shared
	 * VAs, so the unmaps are unconditional; a thread wedged on a
	 * dead-bus read leaks only its private mappings until reboot.
	 */
	f286_unmap_mmio(&rad->waon_e);
	f286_unmap_mmio(&rad->waon_d);
	f286_unmap_mmio(&rad->whale_apb);
	f286_unmap_mmio(&rad->wcfg_e010);
	f286_unmap_mmio(&rad->key);
	f286_unmap_mmio(&rad->aon);
	f286_unmap_mmio(&rad->pmu);
	f286_unmap_mmio(&rad->adi);
}

static const struct of_device_id f286_radio_match[] = {
	{ .compatible = "fplinux,ums9117-f286-radio" },
	{}
};
MODULE_DEVICE_TABLE(of, f286_radio_match);

static struct platform_driver f286_radio_driver = {
	.probe = f286_radio_probe,
	.remove = f286_radio_remove,
	.driver = {
		.name = "ums9117-f286-radio",
		.of_match_table = f286_radio_match,
	},
};
module_platform_driver(f286_radio_driver);

MODULE_DESCRIPTION("Ruio F286 (UMS9117) baseband power-on and DSP bring-up");
MODULE_LICENSE("GPL");
