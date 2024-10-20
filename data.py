

class CycleGANStyleTransferDataset(Dataset):
    def __init__(self, content_dir, style_dir, transform=None):
        """
        Args:
            content_dir (str): Directory path for content images (Domain A).
            style_dir (str): Directory path for style images (Domain B).
            transform (callable, optional): A function/transform to apply to both domains.
        """
        # Filter and load only valid image files (ignoring files like .DS_Store)
        self.content_images = sorted([f for f in os.listdir(content_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
        self.style_images = sorted([f for f in os.listdir(style_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
        self.content_dir = content_dir
        self.style_dir = style_dir
        self.transform = transform

    def __len__(self):
        return min(len(self.content_images), len(self.style_images))

    def __getitem__(self, idx):
      while True:
          content_image_path = os.path.join(self.content_dir, self.content_images[idx % len(self.content_images)])
          try:
              content_image = Image.open(content_image_path).convert('RGB')
              break
          except UnidentifiedImageError:
              print(f"Error: Cannot open content image {content_image_path}. Skipping.")
              idx += 1
      while True:
          style_image_path = os.path.join(self.style_dir, self.style_images[idx % len(self.style_images)])
          try:
              style_image = Image.open(style_image_path).convert('RGB')
              break
          except UnidentifiedImageError:
              print(f"Error: Cannot open style image {style_image_path}. Skipping.")
              idx += 1
      if self.transform:
          content_image = self.transform(content_image)
          style_image = self.transform(style_image)

      return content_image, style_image
