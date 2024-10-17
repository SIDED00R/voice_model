import os
import torch
import soundfile as sf
import librosa
import pandas as pd
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, Trainer, TrainingArguments
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score


# 1. CSV 파일 로드 및 전처리 (None 값 제거)
csv_file_path = '/data/leedominico/repos/gcp_project/labeling.csv'
df = pd.read_csv(csv_file_path).dropna(subset=['감정'])  # None 값 제거

# 감정 레이블 정의 및 매핑
emotion_labels = df['감정'].unique().tolist()
label2id = {str(emotion): int(i) for i, emotion in enumerate(emotion_labels)}
id2label = {int(i): str(emotion) for i, emotion in enumerate(emotion_labels)}

# 2. 음성 파일 로드 함수 정의
def load_audio(file_path):
    #print(f"Trying to load: {file_path}")  # 경로 확인용 로그 추가
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    speech, rate = sf.read(file_path)
    if rate != 16000:  # 16000Hz로 리샘플링
        speech = librosa.resample(speech, orig_sr=rate, target_sr=16000)
    return speech

# 3. SpeechDataset 클래스 정의
class SpeechDataset(Dataset):
    def __init__(self, df, base_directory, folders):
        self.df = df.reset_index(drop=True)  # 인덱스 재설정
        self.base_directory = base_directory
        self.folders = folders

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        folder_idx = (row['연번'] - 1) // 160  # 폴더 결정
        folder_name = self.folders[folder_idx]

        # 파일 번호 형식화
        file_number = row['연번']
        file_name = f"{folder_name}_{file_number:06d}.wav"

        # 파일 경로 설정 (중간에 'wav_48000' 폴더 포함)
        file_path = os.path.join(self.base_directory, folder_name, "wav_48000", file_name)

        # 음성 파일 로드 및 전처리
        speech = load_audio(file_path)
        inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)

        # 감정 레이블 변환
        label = label2id[row['감정']]

        return {
            "input_values": inputs.input_values[0],
            "attention_mask": inputs.attention_mask[0] if "attention_mask" in inputs else None,
            "labels": torch.tensor(label, dtype=torch.long),
        }

# 4. collate_fn 정의: 배치 생성 시 입력을 동일한 길이로 패딩
def collate_fn(batch):
    input_values = [item["input_values"] for item in batch]
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)

    # 입력값 패딩
    input_values_padded = pad_sequence(input_values, batch_first=True, padding_value=0.0)

    # attention_mask 생성 (패딩이 아닌 부분은 1, 나머지는 0)
    attention_mask = (input_values_padded != 0).long()

    return {
        "input_values": input_values_padded,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# 1. 평가 메트릭 함수 정의
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)  # 가장 높은 로짓의 인덱스가 예측 값
    acc = accuracy_score(labels, preds)  # 정확도 계산
    return {"accuracy": acc}

# 5. 데이터셋 및 DataLoader 준비
audio_directory = "/data/leedominico/repos/gcp_project/data/small/"

train_folders = [f"M{2001 + i}" for i in range(8)]  # M2001 ~ M2008
test_folders = [f"M{2009 + i}" for i in range(2)]  # M2009 ~ M2010

train_dataset = SpeechDataset(df, audio_directory, train_folders)
test_dataset = SpeechDataset(df, audio_directory, test_folders)

train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn)

# 6. 모델 및 프로세서 설정
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-large-960h",
    num_labels=len(emotion_labels),
    label2id=label2id,
    id2label=id2label,
    problem_type="single_label_classification"
)

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

# 7. 훈련 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
)

# 8. Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=processor.feature_extractor,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,  # 여기서 평가 메트릭 함수 전달
)
# 9. 모델 학습
trainer.train()

# 10. 테스트 평가
results = trainer.evaluate()
print(f"Test Results: {results}")
