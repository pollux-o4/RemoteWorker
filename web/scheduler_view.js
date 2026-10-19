const messages = {
  clip_required: ['다음 CLIP 우선 처리', '다음 conditioning을 완료한 뒤 VAE를 준비합니다.'],
  clip_window: ['CLIP 처리 구간', '경계 전까지 VAE 웜업을 보류합니다.'],
  sampling_progress: ['샘플링 진행 확인 중', '진행 step을 확인한 뒤 VAE 준비 시점을 결정합니다.'],
  vae_window: ['VAE 준비 구간', '종료 준비 구간 · 미처리 CLIP이 들어오면 먼저 처리합니다.'],
  vae_ready: ['VAE 준비 완료', '대기 작업의 conditioning이 없으면 CLIP을 먼저 처리한 뒤 VAE로 전환합니다.'],
};

export function schedulerText(scheduler) {
  const decision=scheduler?.decision;
  if (!decision?.reason) return null;
  const [title,description]=messages[decision.reason] || ['작업 상태 확인 중',''];
  const progress=scheduler.sampling;
  const remaining=progress?.prompt_id===decision.prompt_id && Number.isFinite(progress?.total) && Number.isFinite(progress?.step)
    ? Math.max(0,progress.total-progress.step) : decision.remaining_steps;
  const timing=Number.isFinite(remaining) ? `샘플링 ${remaining} step 남음` : '';
  return {title,description,timing};
}
