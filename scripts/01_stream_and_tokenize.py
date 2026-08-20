import os
import time
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel


def stream_code_dataset(language_subset: str, max_retries: int = 5):
    data_dir_path = f"data/{language_subset.lower()}"

    for attempt in range(max_retries):
        try:
            print(
                f"Connecting to dataset stream for {data_dir_path} (Attempt {attempt + 1}/{max_retries})..."
            )
            dataset = load_dataset(
                "bigcode/the-stack-smol",
                split="train",
                data_dir=data_dir_path,
                streaming=True,
                token=True,
            )
            return dataset
        except Exception as e:
            print(f"Connection error encountered: {e}")
            if attempt < max_retries - 1:
                sleep_time = 5 * (attempt + 1)
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                raise RuntimeError(
                    f"Failed to connect to Hugging Face after {max_retries} attempts."
                )


def text_iterator(datasets, sample_limit=250000):
    count = 0
    for dataset in datasets:
        iterator = iter(dataset)
        while count < sample_limit:
            try:
                example = next(iterator)
                content = example.get("content")
                if content:
                    yield content
                    count += 1
            except StopIteration:
                break
            except Exception as e:
                # Catch mid-stream connection drops and continue seamlessly
                print(f"Stream interrupted ({e}). Resuming...")
                time.sleep(2)
                continue


def train_custom_tokenizer(output_dir: str):
    python_stream = stream_code_dataset("Python")
    cpp_stream = stream_code_dataset("C++")

    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    special_tokens = [
        "<|unk|>",
        "<|pad|>",
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
    ]

    trainer = BpeTrainer(
        vocab_size=32000, special_tokens=special_tokens, show_progress=True
    )

    print("Training tokenizer...")
    iterator = text_iterator([python_stream, cpp_stream])
    tokenizer.train_from_iterator(iterator, trainer=trainer)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "code-bpe-tokenizer.json")
    tokenizer.save(save_path)
    print(f"Tokenizer successfully trained and saved to {save_path}")


if __name__ == "__main__":
    TOKENIZER_OUT_DIR = os.path.join(
        os.path.dirname(__file__), "../models/base_tokenizer"
    )
    train_custom_tokenizer(TOKENIZER_OUT_DIR)
