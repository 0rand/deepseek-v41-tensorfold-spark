//! The CUDA half of the root build: kernel fatbins with each Python extension's nvcc flags, the runtime, Nemotron, the CLI.

const std = @import("std");

/// Each .cu in zig/kernels/cuda (`src`, else `name`) with the flags its Python extension passes in `extra_cuda_cflags`.
const Kernel = struct { name: []const u8, flags: []const []const u8, src: ?[]const u8 = null, arch_specific: bool = false };

/// The torch-op replacements' qualification flags (runs/006): no contraction, no flush to zero.
const torch_ops = &[_][]const u8{ "-O3", "--fmad=false", "--ftz=false" };

const kernels = [_]Kernel{
    .{ .name = "gdn", .flags = &.{ "-O3", "--fmad=false" } }, // cuda/kernels/gdn.py, tensorfold_gdn_v2
    .{ .name = "probe", .flags = &.{"-O3"} },
    .{ .name = "qmm_group", .flags = &.{"-O3"} }, // cuda/kernels/qmm.py, tensorfold_qmm_v5
    .{ .name = "qmm_prefill", .flags = &.{"-O3"} },
    .{ .name = "experts", .flags = &.{"-O3"} }, // cuda/experts.py, tensorfold_experts_v7
    .{ .name = "experts_prefill", .flags = &.{"-O3"} },
    .{ .name = "experts_pack", .flags = &.{"-O3"} },
    .{ .name = "prefill_attention", .flags = &.{ "-O3", "--fmad=false" } }, // tensorfold_prefill_attention_v1
    .{ .name = "scan_rows", .flags = &.{ "-O3", "--fmad=false" } }, // nemotron_h/cuda/mamba.py
    .{ .name = "nemotron_ops", .flags = &.{"-O3"} }, // ours: Nemotron's layouts and the serial feed
    .{ .name = "lane_gemv", .flags = &.{"-O3"} }, // ours: qmm_group's arithmetic, a column tile's K slices in one CTA
    .{ .name = "sample", .flags = &.{ "-O3", "--fmad=false", "--ftz=false" } }, // ours: the Metal engine's keyed draws
    .{ .name = "torch_argmax", .src = "torch_ops/argmax", .flags = torch_ops },
    .{ .name = "torch_topk", .src = "torch_ops/topk", .flags = torch_ops },
    .{ .name = "torch_pointwise", .src = "torch_ops/pointwise", .flags = torch_ops },
    .{ .name = "torch_indexing", .src = "torch_ops/indexing", .flags = torch_ops },
    .{ .name = "torch_movement", .src = "torch_ops/movement", .flags = torch_ops },
    .{ .name = "torch_nemotron_constants", .src = "torch_ops/nemotron_constants", .flags = torch_ops },
};

/// DeepSeek-V4.1's kernels (zig/kernels/cuda/deepseek_v41, ported from the Python reference),
/// each with its extension's `extra_cuda_cflags`; embedded by the `dsv41_kernels` module.
const dsv41_kernels = [_]Kernel{
    .{ .name = "dsv41_exl3_experts", .src = "deepseek_v41/exl3_experts", .flags = &.{ "-O3", "-lineinfo" } }, // tensorfold_exl3_experts_v1
    .{ .name = "dsv41_x3ld", .src = "deepseek_v41/x3ld", .flags = &.{ "-O3", "-lineinfo" } }, // tf_dsv41_x3ld_v1
    .{ .name = "dsv41_x3pf", .src = "deepseek_v41/x3pf", .flags = &.{ "-O3", "-lineinfo" } }, // tf_dsv41_x3pf_v1
    .{ .name = "dsv41_x3gm", .src = "deepseek_v41/x3gm", .flags = &.{ "-O3", "-lineinfo" } }, // tf_dsv41_x3gm_v1
    .{ .name = "dsv41_x3gm3", .src = "deepseek_v41/x3gm3", .flags = &.{ "-O3", "-lineinfo" } }, // ours: x3gm v3 (x3gm.cu's device code)
    .{ .name = "dsv41_x3ld_epi", .src = "deepseek_v41/x3ld_epi", .flags = &.{ "-O3", "-lineinfo" } }, // ours: x3ld + the expert epilogues (TF_DSV41_X3LD_EPI)
    .{ .name = "dsv41_x3gm_plan", .src = "deepseek_v41/x3gm_plan", .flags = &.{"-O3"} }, // ours: x3gm.plan's torch ops
    .{ .name = "dsv41_topk_keys", .src = "deepseek_v41/topk_keys", .flags = &.{"-O3"} }, // ours: pick.top's torch.topk
    .{ .name = "dsv41_kvsplit", .src = "deepseek_v41/kvsplit", .flags = &.{"-O3"} }, // ours: split KV's exchange copies
    .{ .name = "dsv41_glue", .src = "deepseek_v41/glue", .flags = &.{ "-O3", "--fmad=false" } }, // ours: the window's torch glue
    .{ .name = "dsv41_pfglue", .src = "deepseek_v41/prefill_glue", .flags = &.{ "-O3", "--fmad=false" } }, // ours: the prefill segment's torch glue
    .{ .name = "dsv41_pointwise", .src = "deepseek_v41/pointwise", .flags = &.{ "-O3", "--fmad=false" } }, // ours: prefill MoE's torch pointwise ops
    .{ .name = "dsv41_linear", .src = "deepseek_v41/linear", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } }, // tensorfold_exl3_linear_v4
    .{ .name = "dsv41_x3seg", .src = "deepseek_v41/x3seg", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } }, // tf_dsv41_x3seg_v3
    .{ .name = "dsv41_dense3", .src = "deepseek_v41/dense3", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } }, // tf_dsv41_dense3_v1
    .{ .name = "dsv41_attn", .src = "deepseek_v41/attn_cuda", .flags = &.{ "-O3", "-lineinfo", "--fmad=false" } }, // tf_dsv41_attn_cuda_v1
    .{ .name = "dsv41_topk", .src = "deepseek_v41/topk_cuda", .flags = &.{ "-O3", "-lineinfo", "--fmad=false" } }, // tf_dsv41_attn_cuda_v1
    .{ .name = "dsv41_topk_b", .src = "deepseek_v41/topk_b", .flags = &.{ "-O3", "-lineinfo", "--fmad=false" } }, // ours: topk_cuda.cu row-bounded (TF_DSV41_INDEX_BOUND)
    .{ .name = "dsv41_mhc", .src = "deepseek_v41/mhc_cuda", .flags = &.{ "-O3", "-lineinfo", "--fmad=false" } }, // tf_dsv41_mhc_cuda_v1
    .{ .name = "dsv41_mhc_pf", .src = "deepseek_v41/mhc_pf", .flags = &.{ "-O3", "-lineinfo", "--fmad=false" } }, // tf_dsv41_mhc_pf_v1
    .{ .name = "dsv41_router_gemv", .src = "deepseek_v41/router_gemv", .flags = &.{ "-O3", "-lineinfo" } }, // tf_dsv41_router_gemv_v2
    .{ .name = "dsv41_kv_glue", .src = "deepseek_v41/kv_glue", .flags = &.{ "-O3", "-lineinfo" } }, // KV RMS uses explicit rounded FP64; store preserves Triton FMA
    .{ .name = "dsv41_router_glue", .src = "deepseek_v41/router_glue", .flags = &.{ "-O3", "-lineinfo" } }, // histogram group + upstream rotation; same FMA policy as experts
    .{ .name = "dsv41_pfdense_k8", .src = "deepseek_v41/pfdense_k8", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } }, // tf_dsv41_pfdense_v1
    .{ .name = "dsv41_pfdense_k10", .src = "deepseek_v41/pfdense_k10", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } },
    .{ .name = "dsv41_pfdense_k12", .src = "deepseek_v41/pfdense_k12", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } },
    .{ .name = "dsv41_pfdense_k16", .src = "deepseek_v41/pfdense_k16", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } },
    .{ .name = "dsv41_engram_gate", .src = "deepseek_v41/engram_gate", .flags = &.{ "-O3", "-lineinfo" } }, // tf_dsv41_engram_gate_v1
    .{ .name = "dsv41_l2pace", .src = "deepseek_v41/l2pace", .flags = &.{"-O3"} }, // tf_dsv41_l2pace_v1
    .{ .name = "dsv41_l2pf", .src = "deepseek_v41/l2pf", .flags = &.{"-O3"} }, // tf_dsv41_l2pf_v1 (glm5_next/spark/l2pf.cu)
};

/// The keyed sampler's kernel (sampling.cu: nucleus.row_stats + cand_gather's pack), optional: built with -Dnvcc, taken
/// from -Dfatbins when that dir holds dsv41_sampling.fatbin, else empty (the sampler then refuses to load).
const dsv41_sampling: Kernel = .{ .name = "dsv41_sampling", .src = "deepseek_v41/sampling", .flags = &.{ "-O3", "--fmad=false" } };
var dsv41_sampling_image: ?std.Build.LazyPath = null;

/// The vision tower's kernels (vision.cu, TF_DSV41_IMAGES=native), optional the same way (dsv41_vision.fatbin): an
/// empty image, and native images refuse at boot.
const dsv41_vision: Kernel = .{ .name = "dsv41_vision", .src = "deepseek_v41_vision/vision", .flags = &.{"-O3"} }; // nvcc defaults as torch: fmad on
var dsv41_vision_image: ?std.Build.LazyPath = null;

/// The two-tile dense linears (lin2.cu, ours: TF_DSV41_LIN2), optional the same way (dsv41_lin2.fatbin): an empty
/// image, and the knob refuses at boot.
const dsv41_lin2: Kernel = .{ .name = "dsv41_lin2", .src = "deepseek_v41/lin2", .flags = &.{ "-O3", "--expt-relaxed-constexpr" } };
var dsv41_lin2_image: ?std.Build.LazyPath = null;

/// The DeepSeek-V4.1 kernel module (zig/src/families/deepseek_v41/kernels.zig) over `cuda`; empty `images`: host-only.
pub fn dsv41Kernels(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, images: []const ?std.Build.LazyPath) *std.Build.Module {
    const options = b.addOptions();
    var with = images.len > 0;
    for (images) |i| with = with and i != null;
    options.addOption(bool, "with_kernels", with);
    options.addOption(usize, "dsv41_images", dsv41_kernels.len); // also keeps this options file apart from the runtime's
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/kernels.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addOptions("dsv41_kernel_options", options);
    if (with) for (dsv41_kernels, images) |k, image| m.addAnonymousImport(b.fmt("dsv41_fatbin_{s}", .{k.name}), .{ .root_source_file = image.? });
    const sampling = if (with) dsv41_sampling_image else null;
    m.addAnonymousImport("dsv41_fatbin_dsv41_sampling", .{ .root_source_file = sampling orelse b.addWriteFiles().add("empty.fatbin", "") });
    const vision = if (with) dsv41_vision_image else null;
    m.addAnonymousImport("dsv41_fatbin_dsv41_vision", .{ .root_source_file = vision orelse b.addWriteFiles().add("empty.fatbin", "") });
    const lin2 = if (with) dsv41_lin2_image else null;
    m.addAnonymousImport("dsv41_fatbin_dsv41_lin2", .{ .root_source_file = lin2 orelse b.addWriteFiles().add("empty.fatbin", "") });
    return m;
}

/// torch.utils.cpp_extension's own nvcc flags (torch 2.13): C++20 and which half/bf16 operators the headers define.
const torch_flags = [_][]const u8{
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
    "-std=c++20",
};

/// core/stagger.zig, the segment schedule the Metal and CUDA runners share, as its own module (no imports).
fn stagger(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = target, .optimize = optimize });
}

/// The runtime module for `target`; `with_kernels` false builds it host-only (empty images).
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, images: []const ?std.Build.LazyPath) *std.Build.Module {
    const options = b.addOptions();
    var with = images.len > 0;
    for (images) |i| with = with and i != null;
    options.addOption(bool, "with_kernels", with);
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    cuda.addImport("stagger", stagger(b, target, optimize));
    if (with) for (kernels, images) |k, image| cuda.addAnonymousImport(b.fmt("fatbin_{s}", .{k.name}), .{ .root_source_file = image.? });
    return cuda;
}

fn family(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, draft_ids: *std.Build.Module) struct { core: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, tokenizer: *std.Build.Module } {
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("tokenizer", tokenizer);
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const nemotron = b.createModule(.{ .root_source_file = b.path("zig/src/families/nemotron/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    nemotron.addImport("cuda", cuda);
    nemotron.addImport("core", core);
    nemotron.addImport("lanes", lanes);
    nemotron.addImport("nemotron_draft_ids", draft_ids);
    return .{ .core = core, .lanes = lanes, .nemotron = nemotron, .tokenizer = tokenizer };
}

/// Linux targets: fatbins (-Dnvcc builds them, -Dfatbins embeds prebuilt ones), `tensorfold` and `tf-cuda-test`.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options) void {
    const nvcc = b.option([]const u8, "nvcc", "nvcc (or a wrapper) that builds the CUDA kernel fatbins");
    const prebuilt = b.option([]const u8, "fatbins", "absolute directory of prebuilt <name>.fatbin files to embed");
    const sms = b.option([]const u8, "sm", "SASS targets, comma separated (121; later 120,89)") orelse "121";
    // the compiler's version text is an input of every fatbin, so a new nvcc rebuilds them all
    const version: ?std.Build.LazyPath = if (prebuilt == null and nvcc != null) blk: {
        const run = b.addSystemCommand(&.{ nvcc.?, "--version" });
        run.has_side_effects = true;
        break :blk run.captureStdOut(.{});
    } else null;
    var images: [kernels.len]?std.Build.LazyPath = @splat(null);
    const fatbin_step = b.step("fatbins", "Build and install the CUDA kernel fatbins alone");
    for (kernels, &images) |k, *image| {
        if (prebuilt) |dir| {
            image.* = b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) }));
        } else if (nvcc) |tool| {
            image.* = fatbin(b, tool, version.?, k, sms);
        }
        if (image.*) |file| fatbin_step.dependOn(&b.addInstallFile(file, b.fmt("fatbin/{s}.fatbin", .{k.name})).step);
    }
    const cuda = runtime(b, target, optimize, if (nvcc != null or prebuilt != null) &images else &.{});
    var dsv41_images: [dsv41_kernels.len]?std.Build.LazyPath = @splat(null);
    for (dsv41_kernels, &dsv41_images) |k, *image| {
        if (prebuilt) |dir| {
            image.* = b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) }));
        } else if (nvcc) |tool| {
            image.* = fatbin(b, tool, version.?, k, sms);
        }
        if (image.*) |file| fatbin_step.dependOn(&b.addInstallFile(file, b.fmt("fatbin/{s}.fatbin", .{k.name})).step);
    }
    if (prebuilt) |dir| {
        const file = b.pathJoin(&.{ dir, "dsv41_sampling.fatbin" });
        if (std.Io.Dir.cwd().access(b.graph.io, file, .{})) |_| {
            dsv41_sampling_image = b.graph.cwdRelativePath(file);
        } else |_| {}
    } else if (nvcc) |tool| dsv41_sampling_image = fatbin(b, tool, version.?, dsv41_sampling, sms);
    if (dsv41_sampling_image) |file| fatbin_step.dependOn(&b.addInstallFile(file, "fatbin/dsv41_sampling.fatbin").step);
    if (prebuilt) |dir| {
        const file = b.pathJoin(&.{ dir, "dsv41_vision.fatbin" });
        if (std.Io.Dir.cwd().access(b.graph.io, file, .{})) |_| {
            dsv41_vision_image = b.graph.cwdRelativePath(file);
        } else |_| {}
    } else if (nvcc) |tool| dsv41_vision_image = fatbin(b, tool, version.?, dsv41_vision, sms);
    if (dsv41_vision_image) |file| fatbin_step.dependOn(&b.addInstallFile(file, "fatbin/dsv41_vision.fatbin").step);
    if (prebuilt) |dir| {
        const file = b.pathJoin(&.{ dir, "dsv41_lin2.fatbin" });
        if (std.Io.Dir.cwd().access(b.graph.io, file, .{})) |_| {
            dsv41_lin2_image = b.graph.cwdRelativePath(file);
        } else |_| {}
    } else if (nvcc) |tool| dsv41_lin2_image = fatbin(b, tool, version.?, dsv41_lin2, sms);
    if (dsv41_lin2_image) |file| fatbin_step.dependOn(&b.addInstallFile(file, "fatbin/dsv41_lin2.fatbin").step);
    const dsv41 = dsv41Kernels(b, target, optimize, cuda, if (nvcc != null or prebuilt != null) &dsv41_images else &.{});
    const mods = family(b, target, optimize, cuda, draft_ids);
    const cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cuda_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cli.addImport("cuda", cuda);
    cli.addImport("core", mods.core);
    cli.addImport("lanes", mods.lanes);
    cli.addImport("nemotron", mods.nemotron);
    b.installArtifact(b.addExecutable(.{ .name = "tensorfold", .root_module = cli }));
    const runner = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    runner.addImport("cuda", cuda);
    runner.addImport("lanes", mods.lanes);
    b.installArtifact(b.addExecutable(.{ .name = "tf-cuda-test", .root_module = runner }));
    const dsv41_runner = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/dsv41_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    dsv41_runner.addImport("cuda", cuda);
    dsv41_runner.addImport("dsv41_kernels", dsv41);
    b.installArtifact(b.addExecutable(.{ .name = "tf-dsv41-test", .root_module = dsv41_runner }));
    const dsv41_m1 = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/m1_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    dsv41_m1.addImport("cuda", cuda);
    // the checkpoint reader alone (the family's only use of core): the serving module brings the tokenizer itself
    dsv41_m1.addImport("core", b.createModule(.{ .root_source_file = b.path("zig/src/core/safetensors_root.zig"), .target = target, .optimize = optimize, .link_libc = true }));
    dsv41_m1.addImport("dsv41_kernels", dsv41);
    const m1_tp = @import("tp.zig").module(b, target, optimize, cuda);
    dsv41_m1.addImport("tp", m1_tp); // M2's forward: the TP collectives
    dsv41_m1.addImport("kv", @import("kv.zig").module(b, target, optimize, cuda, m1_tp)); // M5: the paged pool, split KV, sessions
    const dsv41_lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    dsv41_m1.addImport("lanes", dsv41_lanes); // M3: the GPU target's lanes types
    // M4: the served engine (serve_engine.zig) for follow / generate, over the server's engine seam
    const dsv41_api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true });
    dsv41_api.addImport("lanes", dsv41_lanes);
    dsv41_m1.addImport("engine_api", dsv41_api);
    const m1_serve = @import("sampling.zig").serve(b, target, optimize, dsv41_lanes).serve;
    dsv41_m1.addImport("dsv41_serve", m1_serve); // the keyed sampler's Picker
    // TF_DSV41_CALIB=measure's calibration text (calib_gpu.zig, via serve_engine.zig): the serving module's tokenizer
    dsv41_m1.addImport("tokenizer", m1_serve.import_table.get("tokenizer").?);
    @import("grammar.zig").addTo(b, dsv41_m1, target, optimize); // structured output (grammar_gpu.zig): xgrammar's core, the mask PTX
    @import("sampling.zig").gpu(b, target, optimize, cuda, dsv41); // tf-dsv41-samp: the keyed sampler's device steps
    b.installArtifact(b.addExecutable(.{ .name = "tf-dsv41-m1", .root_module = dsv41_m1 }));
    // M4 on the GPU: tensorfold-dsv41, the native server with DeepSeek-V4.1's CUDA engine (rank 0; the other ranks run
    // tf-dsv41-m1 follow)
    const ds_engine = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/serve_engine.zig"), .target = target, .optimize = optimize, .link_libc = true });
    ds_engine.addImport("cuda", cuda);
    ds_engine.addImport("core", b.createModule(.{ .root_source_file = b.path("zig/src/core/safetensors_root.zig"), .target = target, .optimize = optimize, .link_libc = true }));
    ds_engine.addImport("dsv41_kernels", dsv41);
    const engine_tp = @import("tp.zig").module(b, target, optimize, cuda);
    ds_engine.addImport("tp", engine_tp);
    ds_engine.addImport("kv", @import("kv.zig").module(b, target, optimize, cuda, engine_tp));
    @import("grammar.zig").addTo(b, ds_engine, target, optimize); // structured output (grammar_gpu.zig)
    b.installArtifact(b.addExecutable(.{ .name = "tensorfold-dsv41", .root_module = @import("server.zig").withEngine(b, target, .ReleaseSafe, ds_engine, build_options) }));
    _ = nativeServer(b, target, optimize, cuda, mods.lanes, mods.nemotron, mods.tokenizer, build_options, true);
}

/// The CUDA engines a native server opens (native/cuda.zig), over the given runtime and families.
fn engines(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module) struct { api: *std.Build.Module, engines: *std.Build.Module } {
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const mod = b.createModule(.{
        .root_source_file = b.path("zig/src/native/cuda.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "engine_api", .module = api }, .{ .name = "lanes", .module = lanes }, .{ .name = "nemotron", .module = nemotron } },
    });
    return .{ .api = api, .engines = mod };
}

/// `zig build native`: tensorfold-native with the CUDA engines into zig-out/native/bin, as the Metal build makes it.
fn nativeServer(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, tokenizer: *std.Build.Module, build_options: *std.Build.Step.Options, install_native: bool) *std.Build.Step.Compile {
    const m = engines(b, target, optimize, cuda, lanes, nemotron);
    // the HTTP side keeps its safety checks; the engine below it runs at `optimize` (the tokenizer is the family's)
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const json = b.createModule(.{ .root_source_file = b.path("zig/src/server/json.zig"), .target = target, .optimize = .ReleaseSafe });
    const dsv41_serve = @import("server.zig").serveModule(b, target, .ReleaseSafe, json, tokenizer, lanes); // the server's DeepSeek-V4.1 text side
    const exe = b.addExecutable(.{ .name = "tensorfold-native", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = m.api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "json", .module = json }, .{ .name = "dsv41_serve", .module = dsv41_serve }, .{ .name = "native_engines", .module = m.engines }, .{ .name = "checkpoint_cli", .module = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cli.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true, .imports = &.{.{ .name = "native_engines", .module = m.engines }} }) } },
    }) });
    exe.root_module.addOptions("build_options", build_options);
    if (!install_native) {
        exe.root_module.strip = true;
        return exe;
    }
    const install = b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "native/bin" } } });
    b.step("native", "tensorfold-native with the CUDA engines into zig-out/native/bin").dependOn(&install.step);
    return exe;
}

/// Host unit tests of the CUDA runtime, the backend-neutral core, the lane core and the CUDA family (no GPU), on any host.
pub fn hostTests(b: *std.Build, draft_ids: *std.Build.Module, step: *std.Build.Step) void {
    const host = b.graph.host;
    const cuda = runtime(b, host, .debug, &.{});
    const mods = family(b, host, .debug, cuda, draft_ids);
    const native = engines(b, host, .debug, cuda, mods.lanes, mods.nemotron).engines;
    for ([_]*std.Build.Module{ cuda, mods.core, mods.lanes, mods.nemotron, native, stagger(b, host, .debug), dsv41Kernels(b, host, .debug, cuda, &.{}) }) |m| step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    // the DSV4.1 decode graphs' buckets, statics and floor (R4; the cache itself is in the runtime's tests)
    const graphs = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/graphs.zig"), .target = host, .optimize = .debug, .link_libc = true });
    graphs.addImport("cuda", cuda);
    graphs.addImport("tp", @import("tp.zig").module(b, host, .debug, cuda));
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = graphs })).step);
    // DSV4.1 row mode (several slots): rowtab's tables and rowmode's program over the real emitter (block.emit)
    const rows = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/rows_test.zig"), .target = host, .optimize = .debug, .link_libc = true });
    rows.addImport("cuda", cuda);
    rows.addImport("dsv41_kernels", dsv41Kernels(b, host, .debug, cuda, &.{}));
    rows.addImport("tp", @import("tp.zig").module(b, host, .debug, cuda));
    const rows_run = b.addRunArtifact(b.addTest(.{ .root_module = rows }));
    b.step("test-dsv41-rows", "DeepSeek-V4.1 row mode (several slots) on the real emitter, host only").dependOn(&rows_run.step);
    step.dependOn(&rows_run.step);
    // DSV4.1 decode windows past 16 rows (block_wide.zig, TF_DSV41_ROWS_CAP past 16) on the real emitter, against Python's rules
    const wide = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/wide_test.zig"), .target = host, .optimize = .debug, .link_libc = true });
    wide.addImport("cuda", cuda);
    wide.addImport("dsv41_kernels", dsv41Kernels(b, host, .debug, cuda, &.{}));
    wide.addImport("tp", @import("tp.zig").module(b, host, .debug, cuda));
    const wide_run = b.addRunArtifact(b.addTest(.{ .root_module = wide }));
    b.step("test-dsv41-wide", "DeepSeek-V4.1 decode windows past 16 rows on the real emitter, host only").dependOn(&wide_run.step);
    step.dependOn(&wide_run.step);
    // DSV4.1 prod knobs on the Zig side: their readers and the fused dense prefill on the real prefill emitter
    const knobs = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/prod_knobs_test.zig"), .target = host, .optimize = .debug, .link_libc = true });
    knobs.addImport("cuda", cuda);
    knobs.addImport("dsv41_kernels", dsv41Kernels(b, host, .debug, cuda, &.{}));
    knobs.addImport("tp", @import("tp.zig").module(b, host, .debug, cuda));
    const knobs_run = b.addRunArtifact(b.addTest(.{ .root_module = knobs }));
    b.step("test-dsv41-knobs", "DeepSeek-V4.1 prod knobs (fused dense prefill, rows, read-ahead) on the real prefill emitter, host only").dependOn(&knobs_run.step);
    step.dependOn(&knobs_run.step);
    // DSV4.1 DSpark over several slots: the passes' launches, statics and messages on the real emitter
    const mdraft = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/mdraft_test.zig"), .target = host, .optimize = .debug, .link_libc = true });
    mdraft.addImport("cuda", cuda);
    mdraft.addImport("dsv41_kernels", dsv41Kernels(b, host, .debug, cuda, &.{}));
    mdraft.addImport("tp", @import("tp.zig").module(b, host, .debug, cuda));
    // the device chain's staging reads draft/dspark.zig's Params (the lanes core's Sampling)
    mdraft.addImport("lanes", b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = host, .optimize = .debug, .link_libc = true }));
    const mdraft_run = b.addRunArtifact(b.addTest(.{ .root_module = mdraft }));
    b.step("test-dsv41-mdraft", "DeepSeek-V4.1 DSpark drafting over several slots on the real emitter, host only").dependOn(&mdraft_run.step);
    step.dependOn(&mdraft_run.step);
    // DSV4.1 long prompts: the stream top-k and split KV's row-blocked attention on the real prefill emitter
    const longpf = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/longpf_test.zig"), .target = host, .optimize = .debug, .link_libc = true });
    longpf.addImport("cuda", cuda);
    longpf.addImport("dsv41_kernels", dsv41Kernels(b, host, .debug, cuda, &.{}));
    longpf.addImport("tp", @import("tp.zig").module(b, host, .debug, cuda));
    const longpf_run = b.addRunArtifact(b.addTest(.{ .root_module = longpf }));
    b.step("test-dsv41-longpf", "DeepSeek-V4.1 long prompts (stream top-k, split KV's row blocks) on the real prefill emitter, host only").dependOn(&longpf_run.step);
    step.dependOn(&longpf_run.step);
    // DSV4.1 image prompts' buffer roles: the boot check and the plan's image options on the real prefill emitter
    const vplan = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/vision_plan_test.zig"), .target = host, .optimize = .debug, .link_libc = true });
    vplan.addImport("cuda", cuda);
    vplan.addImport("dsv41_kernels", dsv41Kernels(b, host, .debug, cuda, &.{}));
    vplan.addImport("tp", @import("tp.zig").module(b, host, .debug, cuda));
    const vplan_run = b.addRunArtifact(b.addTest(.{ .root_module = vplan }));
    b.step("test-dsv41-vision-plan", "DeepSeek-V4.1 image prompts' buffer roles (the boot check, the plan's options), host only").dependOn(&vplan_run.step);
    step.dependOn(&vplan_run.step);
    // DSV4.1 x3gm v2 on the prefill emitter (TF_DSV41_GM_V2): its reader and gm2_kernel's launches on the real emitter
    const gm2pf = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/gm2pf_test.zig"), .target = host, .optimize = .debug, .link_libc = true });
    gm2pf.addImport("cuda", cuda);
    gm2pf.addImport("dsv41_kernels", dsv41Kernels(b, host, .debug, cuda, &.{}));
    gm2pf.addImport("tp", @import("tp.zig").module(b, host, .debug, cuda));
    const gm2pf_run = b.addRunArtifact(b.addTest(.{ .root_module = gm2pf }));
    b.step("test-dsv41-gm2pf", "DeepSeek-V4.1 x3gm v2 (TF_DSV41_GM_V2) on the real prefill emitter, host only").dependOn(&gm2pf_run.step);
    step.dependOn(&gm2pf_run.step);
    const cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cuda_main.zig"), .target = host, .optimize = .debug, .link_libc = true });
    cli.addImport("cuda", cuda);
    cli.addImport("core", mods.core);
    cli.addImport("lanes", mods.lanes);
    cli.addImport("nemotron", mods.nemotron);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cli })).step);
}

/// nvcc -fatbin with torch's flags, the kernel's own and one -gencode per SASS target, as the Python build passes them.
fn fatbin(b: *std.Build, nvcc: []const u8, version: std.Build.LazyPath, k: Kernel, sms: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ nvcc, "-fatbin" });
    run.addFileInput(version);
    run.addArgs(&torch_flags);
    run.addArgs(k.flags);
    const a = if (k.arch_specific) "a" else "";
    var it = std.mem.tokenizeScalar(u8, sms, ',');
    while (it.next()) |sm| run.addArg(b.fmt("-gencode=arch=compute_{s}{s},code=sm_{s}{s}", .{ sm, a, sm, a }));
    run.addArgs(&.{ "-MD", "-MF" });
    _ = run.addDepFileOutputArg2(b.fmt("{s}.d", .{k.name}), .{});
    run.addArg("-o");
    const out = run.addOutputFileArg(b.fmt("{s}.fatbin", .{k.name}));
    run.addFileArg(b.path(b.fmt("zig/kernels/cuda/{s}.cu", .{k.src orelse k.name})));
    return out;
}

/// Cross-build a release server, embedding the same fatbins on either host CPU.
pub fn distServer(b: *std.Build, target: std.Build.ResolvedTarget, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options, prebuilt: ?[]const u8) *std.Build.Step.Compile {
    var images: [kernels.len]?std.Build.LazyPath = @splat(null);
    if (prebuilt) |dir| for (kernels, &images) |k, *image| {
        image.* = b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) }));
    };
    const cuda = runtime(b, target, .fast, if (prebuilt != null) &images else &.{});
    const mods = family(b, target, .fast, cuda, draft_ids);
    return nativeServer(b, target, .fast, cuda, mods.lanes, mods.nemotron, mods.tokenizer, build_options, false);
}

/// Validate the complete named input set, including images unused by today's server.
pub fn checkDistFatbins(b: *std.Build, dir: []const u8) *std.Build.Step {
    const check = b.addSystemCommand(&.{ "sh", "-c", "for file do [ -f \"$file\" ] && [ -s \"$file\" ] || { echo \"missing or empty CUDA fatbin: $file\" >&2; exit 1; }; done", "check-fatbins" });
    for (kernels) |k| check.addFileArg(b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) })));
    return &check.step;
}
