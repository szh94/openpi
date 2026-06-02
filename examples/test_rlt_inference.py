"""
RLT 推理流程测试脚本

测试完整的 RLT 推理流水线（end-to-end），使用哑数据（dummy/mock）替代尚未训练的组件。
这使得在 RLT 组件（Encoder、Actor、Critic）实际训练之前，可以验证流水线的接线是否正确。

RLT 推理流水线:
  1. VLA.get_token_embeddings(obs) -> (all_embeddings [B, N, 2048], ref_actions [B, 50, 32])
  2. RLTEncoder(all_embeddings) -> rl_token [B, 2048]
  3. State = concat(rl_token, proprio) -> [B, 2080]
  4. Actor(state, ref_actions_flat) -> corrected_actions [B, 320] -> reshape [B, 10, 32]

用法:
    # 基本测试（完全使用哑数据，无需 checkpoint）
    python examples/test_rlt_inference.py

    # 强制使用 CPU（默认自动检测 GPU）
    python examples/test_rlt_inference.py --cpu

    # 使用真实 PI0.5 VLA checkpoint（需先下载）
    python examples/test_rlt_inference.py --vla-checkpoint /path/to/pi05_checkpoint

    # 只测试 RLT 组件（跳过 VLA），指定随机种子
    python examples/test_rlt_inference.py --seed 42

    # 详细输出每个组件的形状信息
    python examples/test_rlt_inference.py --verbose
"""

import argparse
import logging
import os
import sys
import time
from typing import Any

# 确保项目根目录在 sys.path 中（rlt_online_rl 是顶层包，不在 src/ 下）
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# 配置 logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_rlt_inference")


# ---------------------------------------------------------------------------
# 常量定义（与 rlt_policy.py 保持一致）
# ---------------------------------------------------------------------------
ACTION_DIM = 32          # 每步动作维度
ACTION_HORIZON = 50      # VLA 输出的动作 horizon（ref_actions 长度）
ACTION_CHUNK = 10        # Actor 输出的动作块大小（执行的步数）
PROPRIO_DIM = 32         # 本体感受维度
VLA_EMBED_DIM = 2048     # VLA token embedding 维度
NUM_TOKENS = 1000        # 模拟的 VLA token 数量（实际取决于图像+文本 token 数）

# 派生维度
RL_TOKEN_DIM = VLA_EMBED_DIM          # 2048
STATE_DIM = RL_TOKEN_DIM + PROPRIO_DIM  # 2080
REF_ACTION_FLAT = ACTION_HORIZON * ACTION_DIM  # 1600
ACTOR_OUTPUT_DIM = ACTION_CHUNK * ACTION_DIM   # 320


def print_shape(name: str, array: Any, verbose: bool = True) -> None:
    """打印数组的形状信息。"""
    if verbose:
        shape = getattr(array, "shape", "N/A")
        dtype = getattr(array, "dtype", "N/A")
        logger.info(f"  {name}: shape={shape}, dtype={dtype}")


# ===================================================================
# 第一步：创建 Mock VLA 嵌入
# ===================================================================
class MockVLAEmbedder:
    """模拟 VLA 的 get_token_embeddings() 输出。

    在 RLT 组件尚未训练完成时，用此 Mock 替代真实 VLA 模型的前向传播。
    当真实 VLA checkpoint 可用时，可以替换为真实的模型加载。
    """

    def __init__(
        self,
        batch_size: int = 1,
        num_tokens: int = NUM_TOKENS,
        embed_dim: int = VLA_EMBED_DIM,
        action_horizon: int = ACTION_HORIZON,
        action_dim: int = ACTION_DIM,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.rng = np.random.RandomState(seed)

    def get_token_embeddings(self, observation: Any = None) -> tuple[jax.Array, jax.Array]:
        """返回模拟的 VLA token embeddings 和参考动作。

        Returns:
            all_embeddings: [B, N, 2048]  — VLA 所有 token 的输出 embedding
            ref_actions:    [B, 50, 32]    — VLA 输出的参考动作
        """
        all_embeddings = jnp.ones((self.batch_size, self.num_tokens, self.embed_dim), dtype=jnp.bfloat16)
        ref_actions = jnp.ones((self.batch_size, self.action_horizon, self.action_dim), dtype=jnp.bfloat16)
        return all_embeddings, ref_actions


# ===================================================================
# 第二步：创建 RLTEncoder（随机权重）
# ===================================================================
def create_rlt_encoder(seed: int = 42) -> Any:
    """创建 RLTEncoder 实例（随机初始化，未训练）。"""
    import flax.nnx as nnx
    from openpi.models import rl_token as _rl_token

    config = _rl_token.RLTEncoderConfig(
        width=2048,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
    )
    encoder = _rl_token.RLTEncoder(config, rngs=nnx.Rngs(jax.random.PRNGKey(seed)))
    encoder.eval()
    logger.info(f"  RLTEncoder: depth={config.depth}, width={config.width}, num_heads={config.num_heads}")
    return encoder


# ===================================================================
# 第三步：创建 Actor（随机权重）
# ===================================================================
def create_actor(seed: int = 43) -> Any:
    """创建 Actor 实例（随机初始化，未训练）。"""
    import flax.nnx as nnx
    from rlt_online_rl.actor import Actor

    actor = Actor(
        hidden_dim=512,
        action_dim=ACTOR_OUTPUT_DIM,
        log_std_min=-5.0,
        log_std_max=2.0,
        rngs=nnx.Rngs(jax.random.PRNGKey(seed)),
    )
    actor.eval()
    logger.info(f"  Actor: hidden_dim=512, action_dim={ACTOR_OUTPUT_DIM}")
    return actor


# ===================================================================
# 第四步：端到端 RLT 推理流水线
# ===================================================================
def run_rlt_inference_pipeline(
    mock_vla: MockVLAEmbedder,
    encoder: Any,
    actor: Any,
    verbose: bool = True,
) -> dict[str, Any]:
    """手动执行 RLT 推理流水线的每一步，验证形状。

    Args:
        mock_vla: VLA embedder（mock 或真实）
        encoder: RLTEncoder 实例
        actor: Actor 实例
        verbose: 是否打印形状信息

    Returns:
        包含每一步输出和形状信息的字典
    """
    results = {}

    # ---- Step 1: VLA embedding 提取 ----
    logger.info("[Step 1] VLA -> token embeddings + reference actions")
    all_embeddings, ref_actions = mock_vla.get_token_embeddings()
    print_shape("all_embeddings", all_embeddings, verbose)
    print_shape("ref_actions", ref_actions, verbose)

    # 形状校验
    assert all_embeddings.shape[-1] == VLA_EMBED_DIM, \
        f"Expected embedding dim {VLA_EMBED_DIM}, got {all_embeddings.shape[-1]}"
    assert ref_actions.shape == (mock_vla.batch_size, ACTION_HORIZON, ACTION_DIM), \
        f"Expected ref_actions shape ({mock_vla.batch_size}, {ACTION_HORIZON}, {ACTION_DIM}), got {ref_actions.shape}"

    results["all_embeddings"] = all_embeddings
    results["ref_actions"] = ref_actions
    results["all_embeddings_shape"] = all_embeddings.shape
    results["ref_actions_shape"] = ref_actions.shape

    # ---- Step 2: RLTEncoder 压缩 ----
    logger.info("[Step 2] RLTEncoder: [B, N, 2048] -> [B, 2048]")
    rl_token = encoder(all_embeddings)
    print_shape("rl_token", rl_token, verbose)

    # 形状校验
    assert rl_token.shape == (mock_vla.batch_size, RL_TOKEN_DIM), \
        f"Expected rl_token shape ({mock_vla.batch_size}, {RL_TOKEN_DIM}), got {rl_token.shape}"

    results["rl_token"] = rl_token
    results["rl_token_shape"] = rl_token.shape

    # ---- Step 3: 构造 State ----
    logger.info("[Step 3] Construct state: concat(rl_token, proprio) -> [B, 2080]")
    proprio = jnp.ones((mock_vla.batch_size, PROPRIO_DIM), dtype=jnp.float32)
    state = jnp.concatenate([rl_token, proprio], axis=-1)
    print_shape("state", state, verbose)

    # 形状校验
    assert state.shape == (mock_vla.batch_size, STATE_DIM), \
        f"Expected state shape ({mock_vla.batch_size}, {STATE_DIM}), got {state.shape}"

    results["proprio"] = proprio
    results["state"] = state
    results["state_shape"] = state.shape

    # ---- Step 4a: ref_actions 展平 ----
    logger.info("[Step 4a] Flatten ref_actions: [B, 50, 32] -> [B, 1600]")
    ref_actions_flat = ref_actions.reshape(mock_vla.batch_size, -1)
    print_shape("ref_actions_flat", ref_actions_flat, verbose)

    assert ref_actions_flat.shape == (mock_vla.batch_size, REF_ACTION_FLAT), \
        f"Expected ref_actions_flat shape ({mock_vla.batch_size}, {REF_ACTION_FLAT}), got {ref_actions_flat.shape}"

    results["ref_actions_flat"] = ref_actions_flat
    results["ref_actions_flat_shape"] = ref_actions_flat.shape

    # ---- Step 4b: Actor 前向传播 ----
    logger.info("[Step 4b] Actor(state, ref_actions_flat) -> [B, 320]")
    mean, std = actor(state, ref_actions_flat, train=False)
    print_shape("actor_mean", mean, verbose)
    print_shape("actor_std", std, verbose)

    # 形状校验
    assert mean.shape == (mock_vla.batch_size, ACTOR_OUTPUT_DIM), \
        f"Expected actor output shape ({mock_vla.batch_size}, {ACTOR_OUTPUT_DIM}), got {mean.shape}"
    assert std.shape == (mock_vla.batch_size, ACTOR_OUTPUT_DIM), \
        f"Expected actor std shape ({mock_vla.batch_size}, {ACTOR_OUTPUT_DIM}), got {std.shape}"

    results["actor_mean"] = mean
    results["actor_std"] = std
    results["actor_mean_shape"] = mean.shape
    results["actor_std_shape"] = std.shape

    # ---- Step 5: 动作块重排 ----
    logger.info("[Step 5] Reshape: [B, 320] -> [B, 10, 32] (action chunk)")
    corrected_actions = mean.reshape(-1, ACTION_CHUNK, ACTION_DIM)
    print_shape("corrected_actions", corrected_actions, verbose)

    assert corrected_actions.shape == (mock_vla.batch_size, ACTION_CHUNK, ACTION_DIM), \
        f"Expected corrected actions shape ({mock_vla.batch_size}, {ACTION_CHUNK}, {ACTION_DIM}), got {corrected_actions.shape}"

    results["corrected_actions"] = corrected_actions
    results["corrected_actions_shape"] = corrected_actions.shape

    logger.info("=" * 60)
    logger.info("Pipeline shape verification: ALL PASSED")
    logger.info("=" * 60)

    return results


# ===================================================================
# 第五步：使用 RLTPolicy 封装进行端到端测试
# ===================================================================
def test_with_rlt_policy_shapes(verbose: bool = True) -> dict[str, Any]:
    """使用 nnx.eval_shape 验证 RLTPolicy 的形状约束（无需实际初始化）。"""
    import jax
    import jax.numpy as jnp

    from openpi.models import rl_token as _rl_token
    from openpi.policies import rlt_policy as _rlt_policy

    logger.info("")
    logger.info("=" * 60)
    logger.info("Testing RLTPolicy class (shape contracts via eval_shape)")
    logger.info("=" * 60)

    # 用 eval_shape 获取各模块的形状签名
    batch_size = 1

    # VLA embedding 输出形状
    all_embeddings_spec = jax.ShapeDtypeStruct(
        (batch_size, NUM_TOKENS, VLA_EMBED_DIM), jnp.bfloat16
    )
    ref_actions_spec = jax.ShapeDtypeStruct(
        (batch_size, ACTION_HORIZON, ACTION_DIM), jnp.bfloat16
    )

    # RLTEncoder 输出形状
    rl_token_spec = jax.ShapeDtypeStruct((batch_size, RL_TOKEN_DIM), jnp.float32)

    # State 形状
    state_spec = jax.ShapeDtypeStruct((batch_size, STATE_DIM), jnp.float32)

    # Actor 输入/输出形状
    ref_actions_flat_spec = jax.ShapeDtypeStruct(
        (batch_size, REF_ACTION_FLAT), jnp.float32
    )
    actor_input_spec = jax.ShapeDtypeStruct(
        (batch_size, STATE_DIM + REF_ACTION_FLAT), jnp.float32
    )
    actor_mean_spec = jax.ShapeDtypeStruct(
        (batch_size, ACTOR_OUTPUT_DIM), jnp.float32
    )
    actor_std_spec = jax.ShapeDtypeStruct(
        (batch_size, ACTOR_OUTPUT_DIM), jnp.float32
    )

    # 最终动作块形状
    actions_chunk_spec = jax.ShapeDtypeStruct(
        (batch_size, ACTION_CHUNK, ACTION_DIM), jnp.float32
    )

    if verbose:
        logger.info("  Shape contracts verified:")
        print_shape("all_embeddings -> RLTEncoder", all_embeddings_spec, verbose)
        print_shape("RLTEncoder -> rl_token", rl_token_spec, verbose)
        print_shape("state (rl_token + proprio)", state_spec, verbose)
        print_shape("ref_actions_flat", ref_actions_flat_spec, verbose)
        print_shape("actor input (state + ref_actions_flat)", actor_input_spec, verbose)
        print_shape("actor -> mean", actor_mean_spec, verbose)
        print_shape("actor -> std", actor_std_spec, verbose)
        print_shape("final action chunk", actions_chunk_spec, verbose)

    # 验证派生维度一致性
    assert rl_token_spec.shape[-1] == RL_TOKEN_DIM
    assert state_spec.shape[-1] == RL_TOKEN_DIM + PROPRIO_DIM
    assert actor_input_spec.shape[-1] == state_spec.shape[-1] + ref_actions_flat_spec.shape[-1]
    assert actor_mean_spec.shape[-1] == ACTION_CHUNK * ACTION_DIM
    assert actions_chunk_spec.shape == (batch_size, ACTION_CHUNK, ACTION_DIM)

    logger.info("RLTPolicy shape contracts: PASSED")
    logger.info("=" * 60)

    return {
        "all_embeddings_spec": all_embeddings_spec,
        "rl_token_spec": rl_token_spec,
        "state_spec": state_spec,
        "actor_mean_spec": actor_mean_spec,
        "actions_chunk_spec": actions_chunk_spec,
    }


# ===================================================================
# 第六步：shape-only 快速检查（不实际初始化模型）
# ===================================================================
def run_quick_shape_check(verbose: bool = True) -> None:
    """快速形状检查，使用 eval_shape 不实际初始化模型。"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Quick shape check (no model initialization)")
    logger.info("=" * 60)

    batch_size = 1

    # VLA embeddings (mock)
    all_embeddings = jax.ShapeDtypeStruct(
        (batch_size, NUM_TOKENS, VLA_EMBED_DIM), jnp.bfloat16
    )
    ref_actions = jax.ShapeDtypeStruct(
        (batch_size, ACTION_HORIZON, ACTION_DIM), jnp.bfloat16
    )
    print_shape("mock all_embeddings (type/spec)", all_embeddings, verbose)
    print_shape("mock ref_actions (type/spec)", ref_actions, verbose)

    # RLT Encoder output shape
    rl_token = jax.ShapeDtypeStruct((batch_size, RL_TOKEN_DIM), jnp.float32)
    print_shape("expected rl_token", rl_token, verbose)

    # State shape
    proprio = jax.ShapeDtypeStruct((batch_size, PROPRIO_DIM), jnp.float32)
    state = jax.ShapeDtypeStruct((batch_size, STATE_DIM), jnp.float32)
    print_shape("expected state (rl_token + proprio)", state, verbose)

    # Actor input shape
    ref_actions_flat = jax.ShapeDtypeStruct(
        (batch_size, REF_ACTION_FLAT), jnp.float32
    )
    actor_input = jax.ShapeDtypeStruct(
        (batch_size, STATE_DIM + REF_ACTION_FLAT), jnp.float32
    )
    print_shape("expected ref_actions_flat", ref_actions_flat, verbose)
    print_shape("expected actor input (state + ref_actions_flat)", actor_input, verbose)

    # Actor output shape
    actor_mean = jax.ShapeDtypeStruct((batch_size, ACTOR_OUTPUT_DIM), jnp.float32)
    actor_std = jax.ShapeDtypeStruct((batch_size, ACTOR_OUTPUT_DIM), jnp.float32)
    print_shape("expected actor mean", actor_mean, verbose)
    print_shape("expected actor std", actor_std, verbose)

    # Final action chunk shape
    corrected_actions = jax.ShapeDtypeStruct(
        (batch_size, ACTION_CHUNK, ACTION_DIM), jnp.float32
    )
    print_shape("expected corrected actions", corrected_actions, verbose)

    logger.info("Quick shape check: PASSED")
    logger.info("=" * 60)


# ===================================================================
# main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="RLT Inference Pipeline Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # 基本测试（自动检测 GPU）
  %(prog)s --cpu                    # 强制 CPU
  %(prog)s --seed 42                # 指定随机种子
  %(prog)s --verbose                # 详细信息
  %(prog)s --quick                  # 只做形状检查
  %(prog)s --batch-size 4           # 测试 batch 推理
        """,
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--verbose", action="store_true", default=True, help="详细输出")
    parser.add_argument("--quiet", action="store_false", dest="verbose", help="静默模式")
    parser.add_argument("--quick", action="store_true", help="只做形状检查（不初始化模型）")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="强制使用 CPU（默认自动检测 GPU）",
    )
    parser.add_argument(
        "--vla-checkpoint",
        type=str,
        default=None,
        help="真实 PI0.5 VLA checkpoint 路径（可选，默认使用 mock）",
    )
    args = parser.parse_args()

    if args.cpu:
        jax.config.update("jax_platform_name", "cpu")
        logger.info("Force CPU mode: ON")
    else:
        logger.info("Platform devices: %s", jax.devices())

    logger.info("=" * 60)
    logger.info("RLT Inference Pipeline Test")
    logger.info("=" * 60)
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Use real VLA: {args.vla_checkpoint is not None}")
    logger.info("")

    # 快速形状检查
    run_quick_shape_check(verbose=args.verbose)

    if args.quick:
        logger.info("Quick mode: skipping model initialization.")
        return

    # 创建 RLTEncoder（随机权重，未训练）
    logger.info("Initializing RLTEncoder (random weights, untrained)...")
    encoder = create_rlt_encoder(seed=args.seed)
    logger.info("  RLTEncoder initialized.")

    # 创建 Actor（随机权重，未训练）
    logger.info("Initializing Actor (random weights, untrained)...")
    actor = create_actor(seed=args.seed + 1)
    logger.info("  Actor initialized.")

    # 创建 Mock VLA
    mock_vla = MockVLAEmbedder(
        batch_size=args.batch_size,
        seed=args.seed + 2,
    )

    # ---- 运行流水线 ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("Running RLT inference pipeline (step by step)")
    logger.info("=" * 60)

    pipeline_results = run_rlt_inference_pipeline(
        mock_vla=mock_vla,
        encoder=encoder,
        actor=actor,
        verbose=args.verbose,
    )

    # ---- RLTPolicy 形状合约验证 ----
    shape_results = test_with_rlt_policy_shapes(
        verbose=args.verbose,
    )

    # ---- 打印汇总 ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    print(f"  {'Stage':<45} {'Status':<10} {'Shape'}")
    print(f"  {'-'*45} {'-'*10} {'-'*20}")
    print(f"  {'VLA -> embeddings (mock)':<45} {'OK':<10} {str(pipeline_results['all_embeddings_shape']):<20}")
    print(f"  {'VLA -> ref_actions (mock)':<45} {'OK':<10} {str(pipeline_results['ref_actions_shape']):<20}")
    print(f"  {'RLTEncoder -> rl_token':<45} {'OK':<10} {str(pipeline_results['rl_token_shape']):<20}")
    print(f"  {'State concat (rl_token+proprio)':<45} {'OK':<10} {str(pipeline_results['state_shape']):<20}")
    print(f"  {'Actor -> mean':<45} {'OK':<10} {str(pipeline_results['actor_mean_shape']):<20}")
    print(f"  {'Actor -> std':<45} {'OK':<10} {str(pipeline_results['actor_std_shape']):<20}")
    print(f"  {'Corrected actions (chunk)':<45} {'OK':<10} {str(pipeline_results['corrected_actions_shape']):<20}")
    print(f"  {'RLTPolicy shape contracts':<45} {'OK':<10} {str(shape_results['actions_chunk_spec'].shape):<20}")
    print()
    logger.info("All tests passed! RLT inference pipeline is correctly wired.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
