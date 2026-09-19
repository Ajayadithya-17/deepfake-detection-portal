from utils.video_predictor import predict_video


video_path = "test_video.mp4"

result = predict_video(video_path)

print("\n========== VIDEO RESULT ==========")
print("Verdict:", result["verdict"])
print("Fake Probability:", f'{result["fake_probability"]:.2f}%')
print("Real Probability:", f'{result["real_probability"]:.2f}%')
print("Frames Analyzed:", result["frames_analyzed"])
print("Total Frames:", result["total_frames"])
print("==================================")
