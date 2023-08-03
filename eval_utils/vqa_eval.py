"""Evaluate on VQA datasets."""

import os
import json

import torchvision.datasets as datasets

import open_clip
import torch
from clip_benchmark.datasets.builder import image_captions_collate_fn
from clip_benchmark.metrics import zeroshot_retrieval as zsr

from wds_eval import create_model



class VqaDataset():
    def __init__(self, path, transform):
        self.annotations_file = os.path.join(path, f'annotations.json')
        assert os.path.exists(self.annotations_file), f'Did not find {self.annotations_file}'
        self.annotations = json.load(open(self.annotations_file, 'r'))['annotations']

        self.questions_file = os.path.join(path, f'questions.json')
        assert os.path.exists(self.questions_file), f'Did not find {self.questions_file}'
        self.questions = json.load(open(self.questions_file, 'r'))['questions']

        assert len(self.questions) == len(self.annotations)

        self.loader = datasets.folder.default_loader
        self.transform = transform

        self.labels = []
        self.processed_questions = []
        self.image_paths = []

        for annotation, question in zip(self.annotations, self.questions):
            choices = question['multiple_choices']
            answer = annotation['multiple_choice_answer']
            assert len(choices) == 18
            assert answer in choices

            assert question['question_id'] == annotation['question_id']
            assert question['image_id'] == annotation['image_id']
            label = choices.index(answer)
        
            questions = [
                f'Question: {question["question"]} Answer: {choice}.'
                for choice in choices
            ]
            self.processed_questions.append(questions)
            self.labels.append(label)

            image_id = question['image_id']
            image_path = os.path.join(path, 'images', f'abstract_v002_val2015_{image_id:012d}.png')
            self.image_paths.append(image_path)

    def __getitem__(self, index):
        path = self.image_paths[index]
        questions = self.processed_questions[index]
        label = self.labels[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        tokenized_questions = open_clip.tokenize(questions)
        import pdb; pdb.set_trace()

        return {
            'images': sample,
            'labels': label,
            'image_paths': path,
            'texts': tokenized_questions
        }

    def __len__(self):
        return len(self.image_paths)


def evaluate_vqa_dataset(
    task, model_arch, model_path, data_root=None, batch_size=64, num_workers=4
):
    """Evaluate CLIP model on VQA task."""
    model, transform, device = create_model(model_arch, model_path)
    tokenizer = open_clip.get_tokenizer(model_arch)

    dataset = VqaDataset('/admin/home-gamaga/data/vqav1/val', transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Eval
    model.eval()
    with torch.no_grad():
        print('*'*80)
        print('Starting VQA eval')
        correct, count = 0.0, 0.0
        for batch in test_loader:
            inputs, labels = batch['images'].to(model.device), batch['labels'].to(model.device)
            texts = batch['texts'].to(model.device)

            batch_size, num_options, seq_length = texts.shape
            texts = torch.reshape(texts, [batch_size * num_options, seq_length])

            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = torch.reshape(text_features, [batch_size, num_options, -1])

            sims = torch.einsum('bf,bcf->bc', image_features, text_features)
            logits = sims * self.model.logit_scale.exp()
        
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            count += len(logits)

        accuracy = correct / count

    print(f'Accuracy: {100*accuracy:.2f}')

    metrics = {'accuracy': accuracy}
    return metrics