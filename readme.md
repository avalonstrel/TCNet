# DeTrais

### Datasets

##### Shoes / Chairs

- file path

  In `./data`

- parameter `obj`

  `--obj=shoes` or `--obj=chairs`

##### Shoes v2

- file structure

  ```
  - QUML_v2
  	|- ShoeV2_photo
  	|	|- 1135045020.png
  	|	|- ...
  	|
  	|- ShoeV2_sketch
  	|	|- 1135045020_1.png
  	|	|- ...
  	|
  	|- photo_train.txt
  	|- photo_test.txt
  	|- sketch_train.txt
  	|- sketch_test.txt
  ```

- parameter `obj,data_root` (The example path is on 10.88.3.92)

  `--obj=shoes_v2 --data_root=/home/lp_user/dataset/sketch/sbir_qian/QUML_v2`

##### Sketchy

- file structure

  ```
  - sketchy
  	|- 256x256
  	|	|- photo
  	|	|	|- tx_000100000000
  	|	|	|	|- airplane
  	|	|	|	|	|- n02691156_507.jpg
  	|	|	|	|	|- ... (othre photos)
  	|	|	|	|- ... (other categories)
  	|	|	|- tx_(does not matter)
  	|	|	
  	|	|- sketch
  	|	|	|- tx_000100000000
  	|	|	|	|- airplane
  	|	|	|	|	|- n02691156_507-1.jpg
  	|	|	|	|	|- ... (other sketches)
  	|	|	|	|- ... (other categories)
  	|	|	|- tx_(does not matter)
  	|
  	|- info
  	|	|- invalid-ambiguous.txt
  	|	|- invalid-context.txt
  	|	|- invalid-error.txt
  	|	|- invalid-pose.txt
  	|	|- testset.txt
  ```

- parameter `obj,data_root` (The example path is on 10.88.3.92)

  `--obj=sketchy --data_root=/home/lp_user/dataset/sketch/sketchy`



### Demo command

`python train.py --obj=shoes_v2 --data_root=/home/lp_user/dataset/sketch/sbir_qian/QUML_v2 --model_type=densenet --feat_dim=1024 --loss_type='triplet,centre' --loss_ratio='0.13,0.0013' --flag=shoesv2-tpl_ctr-densebn`

- training from pretrained model

  set `--phase=train_continue` and make sure parameters `flag` and `obj` are identical with the previous model.

  