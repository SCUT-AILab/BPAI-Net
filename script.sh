#train and test on Drive
#backbone:resnet50
#train
python main_drive.py --arch fusion --arch_cnn resnet50 --num_segments 8 --xyc --first layer2 --dropout 0.8 --shift --mode train --root_model exp/test --root_log exp/test --tune_from=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth --gcn_pretrained=pretrained/st_gcn.kinetics.pt
#test
python test_drive.py --arch fusion --arch_cnn resnet50 --num_segments 8 --xyc --first layer2 --shift --test_crops=1 --batch-size=8 --mode test --model_path exp/test/checkpoint.best.pth --root_log exp/test/

#backbone:mobilenetv2
#train
python main_drive.py --arch fusion --arch_cnn mobilenetv2 --num_segments 8 --xyc --first 4 --dropout 0.5 --shift --mode train --root_model exp/test --root_log exp/test --tune_from=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth --gcn_pretrained=pretrained/st_gcn.kinetics.pt
#test
python test_drive.py --arch fusion --arch_cnn mobilenetv2 --num_segments 8 --xyc --first 4 --shift --test_crops=1 --batch-size=8 --mode test --model_path exp/test/checkpoint.best.pth --root_log exp/test/

#backbone:I3d
train
python main_drive.py --arch i3d_all --num_segments 64 --xyc --first 3c --dropout 0.8 --shift --mode train --root_model exp/test --root_log exp/test --tune_from=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth --gcn_pretrained=pretrained/st_gcn.kinetics.pt
#test
python test_drive.py --arch i3d_all --num_segments 64 --xyc --first 3c --shift --test_crops=1 --batch-size=8 --mode test --model_path exp/test/checkpoint.best.pth --root_log exp/test/



#train and test on PCL-BDB
#backbone:resnet50
#train
python main_drive.py --dataset pcl --arch fusion --arch_cnn resnet50 --gpus 1 --num_class 40 --num_segments 8 --first layer2 --xyc --batch-size 8 --dropout 0.8 --shift --mode train --root_model exp/test --root_log exp/test --root dataset/pcl/ --skeleton_json dataset/pcl/video_pose --tune_from=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth --gcn_pretrained=pretrained/st_gcn.kinetics.pt --pcl_anno annotation(2)(1).json
#test
python test_drive.py --dataset pcl --arch fusion --arch_cnn resnet50 --num_segments 8 --num_class 40 --first layer2 --xyc --test_crops=1 --batch-size=8 --mode test --model_path exp/test/checkpoint.best.pth --root_log exp/test --pcl_anno annotation(2)(1).json --root dataset/pcl/ --skeleton_json dataset/pcl/video_pose

#backbone:mobilenetv2
#train
python main_drive.py --dataset pcl --arch fusion --arch_cnn mobilenetv2 --gpus 0 --num_class 40 --num_segments 8 --first 4 --xyc --batch-size 8 --dropout 0.5 --shift --mode train --root_model exp/test --root_log exp/test --root dataset/pcl/ --skeleton_json dataset/pcl/video_pose --tune_from=pretrained/TSM_kinetics_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100_dense.pth --gcn_pretrained=pretrained/st_gcn.kinetics.pt --pcl_anno annotation(2)(1).json
#test
python test_drive.py --dataset pcl --arch fusion --arch_cnn mobilenetv2 --num_segments 8 --num_class 40 --first 4 --xyc --test_crops=1 --batch-size=8 --mode test --model_path exp/test/checkpoint.best.pth --root_log exp/test --pcl_anno annotation(2)(1).json --root dataset/pcl/ --skeleton_json dataset/pcl/video_pose