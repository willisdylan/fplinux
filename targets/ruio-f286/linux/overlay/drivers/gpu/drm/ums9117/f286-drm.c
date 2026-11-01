// SPDX-License-Identifier: GPL-2.0-only
#include <linux/bitops.h>
#include <linux/module.h>
#include <linux/of.h>
#include <linux/platform_device.h>

#include "ums9117-drm.h"

#include "fplinux-ruio-f286-panel.h"

static const struct ums9117_drm_profile f286_drm_profile = {
	.name = "ruio-f286-rgb565",
	.transport = FPLINUX_PANEL_TRANSPORT,
	.init = fplinux_panel_init,
	.init_count = ARRAY_SIZE(fplinux_panel_init),
	.init_finish = fplinux_panel_cold_finish,
	.init_finish_count = ARRAY_SIZE(fplinux_panel_cold_finish),
	.wake_finish = fplinux_panel_wake_finish,
	.wake_finish_count = ARRAY_SIZE(fplinux_panel_wake_finish),
	.width = FPLINUX_PANEL_WIDTH,
	.height = FPLINUX_PANEL_HEIGHT,
	.reset_phase_ms = 30,
	.reset_release_ms = 0,
	.wake_reset_phase_ms = 30,
	.display_off_ms = 120,
	.sleep_in_ms = 50,
	.sleep_out_ms = 120,
	.wled_backlight_name = "f286-backlight",
	.lcdc_ctrl_set = BIT(1),
	.lcdc_ctrl_clear = BIT(2) | (7U << 5),
};

static const struct of_device_id f286_drm_of_match[] = {
	{ .compatible = "fplinux,ums9117-gc9307-lcm-fb",
	  .data = &f286_drm_profile },
	{}
};
MODULE_DEVICE_TABLE(of, f286_drm_of_match);

static struct platform_driver f286_drm_driver = {
	.probe = ums9117_drm_probe,
	.remove = ums9117_drm_remove,
	.shutdown = ums9117_drm_shutdown,
	.driver = {
		.name = "ruio-f286-drm",
		.of_match_table = f286_drm_of_match,
		.pm = pm_sleep_ptr(&ums9117_drm_pm_ops),
	},
};
module_platform_driver(f286_drm_driver);

MODULE_DESCRIPTION("Ruio F286 GC9307 panel profile");
MODULE_LICENSE("GPL");
