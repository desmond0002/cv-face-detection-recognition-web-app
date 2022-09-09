import os

path = r"/home/andrew/diploma/flask/all_classes/5_classes/val_dir/putin"

i = 1

for file_name in os.listdir(path):
    # Имя файла и его формат
    base_name, ext = os.path.splitext(file_name)

    # Нужны файлы определенного формата
    if ext.lower() not in ['.jpg', '.png']:
        continue

    # Полный путь к текущему файлу
    abs_file_name = os.path.join(path, file_name)

    # Полный путь к текущему файлу с новым названием
    new_abs_file_name = os.path.join(path, "putin" + str(i) + ext)

    os.rename(abs_file_name, new_abs_file_name)

    i += 1
