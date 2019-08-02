# -*- coding: utf-8 -*-
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='the batch size, default to 128')
    parser.add_argument('--epoches', type=int, default=50, help='the epoches, default to 50')
    parser.add_argument('--predict', action="store_true", help="predict", default=False)
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    a = parser.parse_args()

    gan = Gan(get_gpus())
    if a.predict:
        gan.predict_gan(a.batch_size)
    else:
        gan.train(lr=a.lr, batch_size=a.batch_size, epoches=a.epoches)