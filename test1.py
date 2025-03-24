import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. 오디오 불러오기
y, sr = librosa.load("your_audio.wav", sr=22050)  # 파일 이름 바꿔줘!

# 2. 파라미터 설정
n_mfcc = 13
hop_length = 512

# 3. MFCC & ZCR 추출
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)

# 4. MFCC + ZCR 결합
features = np.vstack([mfcc, zcr])          # (14, N프레임)
features = features.T                      # (N프레임, 14)
columns = [f'MFCC_{i+1}' for i in range(n_mfcc)] + ['ZCR']
df = pd.DataFrame(features, columns=columns)

# 5. CSV로 저장
df.to_csv("mfcc_zcr_features.csv", index=False)
print("CSV 파일 저장 완료!")

# 6. 시각화 (MFCC)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis='time', sr=sr, hop_length=hop_length)
plt.colorbar(format='%+2.0f dB')
plt.title('MFCC')
plt.tight_layout()
plt.show()

# 7. 시각화 (ZCR)
plt.figure(figsize=(10, 3))
plt.plot(zcr[0])
plt.title("Zero Crossing Rate")
plt.xlabel("Frame")
plt.ylabel("ZCR")
plt.grid()
plt.tight_layout()
plt.show()
