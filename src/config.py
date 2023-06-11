import torch


class Settings:
    def __init__(self,
                 task: str = 'text',
                 algo: str = 'Discop',
                 model_name: str = 'gpt2',
                 temp: float = 1.0,
                 top_p: float = 0.92,
                 length: int = 100,
                 seed: int = 100,
                 device=torch.device('cpu')):
        self.task = task
        # assert task in ['text, 'image', 'text-to-speech']
        self.algo = algo
        # assert algo in ['Discop', 'Discop_baseline', 'sample']
        self.model_name = model_name
        self.temp = temp
        self.top_p = top_p
        self.length = length
        self.seed = seed
        self.device = device

    def __call__(self):
        return self.algo, self.temp, self.top_p, self.length, self.seed

    def __str__(self):
        return '\n'.join('{} = {}'.format(key, value) for (key, value) in self.__dict__.items())


# text_default_settings = Settings('text', model_name='gpt2', top_p=0.8, length=100)
text_default_settings = Settings('text', model_name='transfo-xl-wt103', top_p=0.8, length=100)

image_default_settings = Settings('image', model_name='openai/imagegpt-small', top_p=1.0)
audio_default_settings = Settings('text-to-speech', model_name='univoc', top_p=0.98)
