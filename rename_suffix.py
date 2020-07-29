from settings import train_path
from Function_API import Image_Processing


Image_Processing.rename_suffix(Image_Processing.extraction_image(train_path))
