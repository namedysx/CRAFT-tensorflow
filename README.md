A implement of paper ：https://arxiv.org/abs/1904.01941

useage
train:
Check your syntext dataset dir in craft.py, and change train(False) to train(True) at last line. 
Run craft.py
text:
modify it

if __name__ == "__main__":
    train(False)
    # test('/home/user4/ysx/demo/CRAFT_214000.ckpt', '/home/user4/ysx/CRAFT/802.jpg')

TO

if __name__ == "__main__":
    # train(False)
    test('/home/user4/ysx/demo/CRAFT_214000.ckpt', '/home/user4/ysx/CRAFT/802.jpg')

the first argument is your .ckpt path, seconed argument is text image path

Fine-tuning:

TO DO

if you want to do this, you could use train(False), it means you load a checkpoint then training

Weak supervision:

TO DO
