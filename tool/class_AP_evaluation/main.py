from BoundingBoxes import *
from BoundingBox import *
from class_AP import *
import json

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    

def main(ground_truth_file, predict_file):
    with open(ground_truth_file, "r") as file:
        ground_truth = json.load(file)
    with open(predict_file, "r") as file:
        predict = json.load(file)
    
    bboxes = BoundingBoxes()
    for truth in ground_truth['annotations']:
        bbox = BoundingBox(imageName=truth['image_id'],
                    classId=truth['category_id'],
                    x=truth['bbox'][0],
                    y=truth['bbox'][1],
                    w=truth['bbox'][2],
                    h=truth['bbox'][3],
                    typeCoordinates=CoordinatesType.Absolute,
                    bbType=BBType.GroundTruth,
                    format=BBFormat.XYWH
                )
        bboxes.addBoundingBox(bbox)
        
    for pred in predict['annotations']:
        bbox = BoundingBox(imageName=pred['image_id'],
                    classId=pred['category_id'],
                    x=pred['bbox'][0],
                    y=pred['bbox'][1],
                    w=pred['bbox'][2],
                    h=pred['bbox'][3],
                    typeCoordinates=CoordinatesType.Absolute,
                    bbType=BBType.Detected,
                    classConfidence=pred['score'],
                    format=BBFormat.XYWH
                )
        bboxes.addBoundingBox(bbox)
    
    E = Evaluator()
    AP = E.GetPascalVOCMetrics(bboxes,
                            IOUThreshold=0.5,
                            method=MethodAveragePrecision.EveryPointInterpolation)
    
    mean_f1 = 0
    
    print(f'{" ":16} {"class AP":6} {"TP":^6} {"FP":^6} {"Precision"}  {"Recall":6} {"F1":^6}')
    print('-'*65)
    for ap in AP:
        precision = ap["total TP"]/(ap["total FP"]+ap["total TP"])
        recall = ap["total TP"]/ap["total positives"]
        f1 = 2 * (precision*recall) / (precision+recall) if precision + recall != 0 else 0
        mean_f1 += f1
        print(f'{classes[ap["class"]]:>15}  {ap["AP"]:6.04f} {ap["total TP"]:6.0f} {ap["total FP"]:6.0f} {precision:11.4f} {recall:7.04f} {f1:7.04f}')
    print('='*65)
    print(f'{"Average":>15}  {" ":40} {mean_f1/len(AP):7.04f}')
        
    print()
        
        
    
if __name__ == '__main__':
    main(
        ground_truth_file = './data/split_valid.json',
        predict_file = './data/predict_valid.json'
    )