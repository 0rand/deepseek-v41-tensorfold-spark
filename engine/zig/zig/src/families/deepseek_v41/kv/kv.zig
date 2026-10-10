//! DeepSeek-V4.1's KV memory on the Zig engine: its cache families, split KV's row exchange, the KV side of the priced floor, the device pool.
//! The family-neutral parts (pool book, page stores, sessions, NVMe tier) are in zig/src/sessions.

pub const sessions = @import("sessions");
pub const layout = @import("layout.zig");
pub const Layout = layout.Layout;
pub const split = @import("split.zig");
pub const budget = @import("budget.zig");
pub const device = @import("device.zig");
pub const DevicePool = device.DevicePool;
pub const sess = @import("sess.zig");
pub const sched = @import("sched.zig");
pub const progress = @import("progress.zig");

test {
    _ = layout;
    _ = split;
    _ = budget;
    _ = device;
    _ = @import("split_test.zig");
    _ = @import("union_plan_test.zig");
    _ = sess;
    _ = @import("sess4_test.zig");
    _ = @import("resend_test.zig");
    _ = sched;
    _ = progress;
}
