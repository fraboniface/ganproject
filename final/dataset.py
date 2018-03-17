import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class PaintingsDataset(Dataset):
    """Dataset class to handle both genre and style label"""
    def __init__(self, csv_file, root_dir, min_samples_class=1000, transform=None):

        self.root_dir = root_dir
        self.transform = transform

        self.data = pd.read_csv(csv_file, index_col=0)
        self.min_samples_class = min_samples_class
        self.get_dataframe()
        
        self.genres = pd.unique(self.data['genre'])
        self.genres.sort()
        self.genre_to_idx = {self.genres[i]: i for i in range(len(self.genres))}
        self.samples_per_genre = [(self.data['genre']==g).sum() for g in self.genres]
        
        self.styles = pd.unique(self.data['style'])
        self.styles.sort()
        self.style_to_idx = {self.styles[i]: i for i in range(len(self.styles))}
        self.samples_per_style = [(self.data['style']==s).sum() for s in self.styles]
        
        self.genre_to_folder = {
            'portrait': 'portraits',
            'landscape': 'landscapes',
            'cityscape': 'cityscapes',
            'genre painting': 'genre_paintings',
            'religious painting': 'religious_paintings'
        }
        
        print(self.data.head())

    def get_dataframe(self):

        available_files = []
        for t in os.walk(self.root_dir):
            available_files += t[2]

        self.data = self.data[self.data.filename.isin(available_files)]
        
        genre = self.data.groupby('genre').count()['filename']
        style = self.data.groupby('style').count()['filename']

        def lookup(df, s, col):
            return s.loc[df[col]]

        self.data['genre_count'] = self.data.apply(lookup, axis=1, args=(genre, 'genre'))
        self.data['style_count'] = self.data.apply(lookup, axis=1, args=(style, 'style'))
        
        self.data = self.data[(self.data.genre_count >= self.min_samples_class) & (self.data.style_count >= self.min_samples_class)]
        self.data.reset_index(drop=True, inplace=True)
     
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        genre = self.data.loc[idx, 'genre']
        img_name = os.path.join(self.root_dir, self.genre_to_folder[genre], self.data.loc[idx, 'filename'])
        genre = self.genre_to_idx[genre]
        
        style = self.data.loc[idx, 'style']
        style = self.style_to_idx[style]
        
        with open(img_name, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, genre, style