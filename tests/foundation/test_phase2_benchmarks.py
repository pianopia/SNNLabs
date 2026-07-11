from src.dst_snn.foundation.phase2_benchmarks import (
    run_image_text_retrieval_benchmark,
    run_text_next_token_benchmark,
)


def test_text_next_token_minimum_benchmark_learns():
    result = run_text_next_token_benchmark(
        samples=24,
        length=5,
        vocab_size=8,
        dim=8,
        teacher_epochs=8,
        student_epochs=10,
        seed=7,
    )
    assert result.final_loss < result.first_loss
    assert result.final_student_score > result.initial_student_score
    assert 0.0 <= result.spike_rate <= 1.0


def test_image_text_minimum_benchmark_learns_retrieval():
    result = run_image_text_retrieval_benchmark(
        pairs=6,
        image_dim=8,
        text_dim=7,
        embed_dim=8,
        epochs=14,
        seed=5,
    )
    assert result.final_loss < result.first_loss
    assert result.final_student_score >= result.initial_student_score
    assert result.teacher_score == 1.0
