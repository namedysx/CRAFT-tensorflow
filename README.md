A implement of paper ：https://arxiv.org/abs/1904.01941

This is the result of training 200,000 steps.

![image](https://github.com/namedysx/CRAFT-tensorflow/blob/master/image/image/t.jpg)

Result

![image](https://github.com/namedysx/CRAFT-tensorflow/blob/master/image/image/weight.jpg)
![image](https://github.com/namedysx/CRAFT-tensorflow/blob/master/image/image/weight_aff.jpg)
![image](https://github.com/namedysx/CRAFT-tensorflow/blob/master/image/image/res_text_image_word.jpg)
![image](https://github.com/namedysx/CRAFT-tensorflow/blob/master/image/image/res_text_image_char.jpg)


Useage

Train:
Check your syntext dataset dir in craft.py, and change train(False) to train(True) at last line. 
Run craft.py

text:

modify below:


if __name__ == "__main__":

    train(False)
    # test('ckpt_path', 'text_image_path')

TO

if __name__ == "__main__":

    # train(False)
    test('ckpt_path', 'text_image_path')

The first argument is your .ckpt path, seconed argument is text image path.

Fine-tuning:

TO DO

If you want to do this, you could use train(False), it means you load a checkpoint then training

Weak supervision:

TO DO
