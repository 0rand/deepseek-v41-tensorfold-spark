//! DeepSeek-V4.1-Flash on the Zig engine: the release config, the EXL3 pack read by its headers, a rank's TP weight plan (host side; kernels in kernels*.zig).

pub const config = @import("config.zig");
pub const Config = config.Config;
pub const exl3 = @import("exl3.zig");
pub const names = @import("names.zig");
pub const pack = @import("pack.zig");
pub const Pack = pack.Pack;
pub const plan = @import("plan.zig");
pub const Plan = plan.Plan;
pub const stage = @import("stage.zig");
pub const named = @import("named.zig");
pub const rounds = @import("rounds.zig");
pub const prepare = @import("prepare.zig");
pub const engram_host = @import("engram_host.zig");
pub const buffers = @import("buffers.zig");

test {
    _ = config;
    _ = exl3;
    _ = names;
    _ = pack;
    _ = plan;
    _ = stage;
    _ = named;
    _ = rounds;
    _ = prepare;
    _ = engram_host;
    _ = buffers;
    _ = @import("host_test.zig");
    // the CUDA port: Engram's shard rows (R3), greedy's host step, the phase timers
    _ = @import("engram_rows_test.zig");
    _ = @import("greedy.zig");
    _ = @import("phases.zig");
    _ = @import("l2pf.zig");
    _ = @import("engram_gate.zig");
    // the CUDA port: the round's launches (TF_DSV41_ROUND_GRAPH)
    _ = @import("round_graph.zig");
    // the CUDA port: prepared weight folders (TF_DSV41_PREPARED)
    _ = @import("prepared.zig");
    _ = @import("boot_mem.zig");
}
