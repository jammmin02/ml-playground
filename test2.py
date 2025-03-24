import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 오디오 로드
y, sr = librosa.load("your_audio.wav", sr=22050)
n_mfcc = 13
hop_length = 512

# 2. 특징 추출
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)

# 3. 합치기
features = np.vstack([mfcc, zcr])        # (14, 프레임 수)
features = features.T                    # (프레임 수, 14)
columns = [f'MFCC_{i+1}' for i in range(n_mfcc)] + ['ZCR']
df = pd.DataFrame(features, columns=columns)

# 4. 저장
df.to_csv("combined_mfcc_zcr.csv", index=False)
print("CSV 저장 완료: combined_mfcc_zcr.csv")

# 5. 시각화 1: MFCC
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis='time', sr=sr, hop_length=hop_length)
plt.colorbar(format='%+2.0f dB')
plt.title('MFCC')
plt.tight_layout()
plt.show()

# 6. 시각화 2: ZCR
plt.figure(figsize=(10, 3))
plt.plot(zcr[0])
plt.title("Zero Crossing Rate")
plt.xlabel("Frame")
plt.ylabel("ZCR")
plt.grid()
plt.tight_layout()
plt.show()

# 7. 시각화 3: 합쳐진 전체 히트맵 (MFCC + ZCR)
plt.figure(figsize=(10, 5))
plt.imshow(features.T, aspect='auto', origin='lower', cmap='coolwarm')
plt.colorbar()
plt.title("Combined Features (MFCC + ZCR)")
plt.xlabel("Frame")
plt.ylabel("Feature Index (MFCC1~13 + ZCR)")
plt.tight_layout()
plt.show()
