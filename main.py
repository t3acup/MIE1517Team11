import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
from ultralytics import YOLO
import kagglehub
import pandas as pd
import requests
import os
import ast
from difflib import get_close_matches
from IPython.display import Markdown
from PIL import Image
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def yolo_predict(model, img):
    """Single-class prediction with box filtering"""
    results = model.predict(img, imgsz=512)

    # Extract predictions
    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        return img, [], [], []

    scores = results[0].boxes.conf.cpu().numpy()

    # Sort by area (descending)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sort_idx = np.argsort(areas)[::-1]

    sorted_boxes = boxes[sort_idx]
    sorted_scores = scores[sort_idx]

    # Filter containing boxes
    keep = []
    suppressed = set()

    for i in range(len(sorted_boxes)):
        if i in suppressed:
            continue
        keep.append(i)
        for j in range(i + 1, len(sorted_boxes)):
            if j in suppressed:
                continue
            xi1, yi1, xi2, yi2 = sorted_boxes[i]
            xj1, yj1, xj2, yj2 = sorted_boxes[j]

            if (xj1 >= xi1) and (yj1 >= yi1) and (xj2 <= xi2) and (yj2 <= yi2):
                suppressed.add(i)
                break

    # Apply filtering
    final_idx = [idx for idx in keep if idx not in suppressed]
    filtered_boxes = sorted_boxes[final_idx]
    filtered_scores = sorted_scores[final_idx]

    # Create annotated image
    output_img = img.copy()
    for box in filtered_boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Crop detected foods
    cropped_images = [img[y1:y2, x1:x2] for x1, y1, x2, y2 in filtered_boxes.astype(int)]

    return output_img, cropped_images, filtered_boxes, filtered_scores

def get_yolo_result(model, image, output_dir=None):
    """Simplified single-class result handler"""
    annotated_img, cropped_images, boxes, scores = yolo_predict(model, image)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, f"annotated.jpg"), annotated_img)

        for i, crop in enumerate(cropped_images):
            cv2.imwrite(os.path.join(output_dir, f"food_{i + 1}.jpg"), crop)

    return {
        'annotated_image': annotated_img,
        'food_images': cropped_images,
        'boxes': boxes,
        'scores': scores
    }

def find_best_match(dish_name, recipe_names):
    matches = get_close_matches(dish_name.lower(), recipe_names, n=1, cutoff=0.6)
    return matches[0] if matches else None

def parse_list_column(column):
    try:
        return ast.literal_eval(column)
    except:
        return []

def clean_output(text, format="plain"):
    text = text.strip()
    if format == "plain":
        text = text.replace("**", "")
        text = text.replace("*", "")
        text = text.replace("\\n", "\n")
        text = text.replace("\n\n", "\n")
    return text

def generate_recipe(dish_names, format="markdown"):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    if isinstance(dish_names, str):
        dish_names = [dish_names]
        single_input = True
    else:
        single_input = False

    results = {}

    for dish_name in dish_names:
        all_names = recipes_df['Name'].dropna().str.lower().tolist()
        matched_name = find_best_match(dish_name, all_names)

        if matched_name:
            matched_recipe = recipes_df[recipes_df['Name'].str.lower() == matched_name].iloc[0]

            ingredients = parse_list_column(matched_recipe['RecipeIngredientParts'])
            steps = parse_list_column(matched_recipe['RecipeInstructions'])

            if len(ingredients) == 0 or len(steps) == 0:
                matched_name = None
            else:
                ingredients_text = "\n".join(f"- {i}" for i in ingredients)
                steps_text = "\n".join(f"{i+1}. {clean_step_text(s)}" for i, s in enumerate(steps))

                nutrition_text = (
                    f"- Calories: {matched_recipe['Calories']} kcal\n"
                    f"- Fat: {matched_recipe['FatContent']} g\n"
                    f"- Protein: {matched_recipe['ProteinContent']} g\n"
                    f"- Carbs: {matched_recipe['CarbohydrateContent']} g\n"
                    f"- Sugar: {matched_recipe['SugarContent']} g\n"
                    f"- Sodium: {matched_recipe['SodiumContent']} mg"
                )

                prep_time = parse_iso_time(matched_recipe.get('PrepTime', ''))
                cook_time = parse_iso_time(matched_recipe.get('CookTime', ''))
                total_time = parse_iso_time(matched_recipe.get("TotalTime", ""))

                # Ask AI for Difficulty
                difficulty_prompt = f"""
Given the recipe name, ingredients, and cooking steps below, estimate the difficulty as one of: Easy, Medium, Hard. Just respond with one word.

Dish Name: {matched_recipe['Name']}

Ingredients:
{ingredients_text}

Cooking Steps:
{steps_text}
"""
                payload_diff = {
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": difficulty_prompt}],
                    "temperature": 0.5
                }
                diff_resp = requests.post(url, headers=headers, json=payload_diff)
                if diff_resp.status_code == 200:
                    difficulty = diff_resp.json()["choices"][0]["message"]["content"].strip().split()[0]
                else:
                    difficulty = "Not specified"

                # Ask AI for Allergen + Dietary Labels
                prompt_for_extra = f"""
Given the following recipe, generate ONLY:

### 5. Allergen Warnings
### 6. Dietary Labels (e.g. Vegan, Gluten-Free, Halal)

Dish Name: {matched_recipe['Name']}

Ingredients:
{ingredients_text}

Nutrition Facts (Per Serving):
{nutrition_text}

Cooking Steps:
{steps_text}
"""
                payload_extra = {
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt_for_extra}],
                    "temperature": 0.7
                }
                response = requests.post(url, headers=headers, json=payload_extra)
                if response.status_code == 200:
                    extra_info = response.json()["choices"][0]["message"]["content"].strip()
                else:
                    extra_info = "### 5. Allergen Warnings\nNot specified\n\n### 6. Dietary Labels\nNot specified"

                full_text = f"""
**Source:** Dataset

## Dish Name: {matched_recipe['Name']}

### 1. Time & Difficulty
- Prep Time: {prep_time}
- Cook Time: {cook_time}
- Total Time: {total_time}
- Difficulty: {difficulty}

### 2. Nutrition Facts (Per Serving)
{nutrition_text}

### 3. Ingredients
{ingredients_text}

### 4. Cooking Steps
{steps_text}

{extra_info}
"""
                results[dish_name] = clean_output(full_text, format=format)
                continue

        # Fallback to AI
        fallback_prompt = f"""
This recipe is AI-generated.

Please generate a clean, markdown-formatted recipe for:

Dish Name: {dish_name}

Your output must follow this structured markdown format:

## Dish Name: <name>

### 1. Time & Difficulty
- Prep Time:
- Cook Time:
- Total Time:
- Difficulty: Easy / Medium / Hard

### 2. Nutrition Facts (Per Serving)
- Calories:
- Protein (g):
- Carbohydrates (g):
- Fat (g):

### 3. Ingredients
- item 1
- item 2
...

### 4. Cooking Steps
1. Step one
2. Step two
...

### 5. Allergen Warnings
...

### 6. Dietary Labels
...

Important:
- Do not skip section 4. Cooking Steps
- Do not include notes or suggestions
- Start directly with “## Dish Name:”
"""

        for _ in range(2):
            payload = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": fallback_prompt}],
                "temperature": 0.7
            }
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                raw_text = response.json()["choices"][0]["message"]["content"]
                if "### 4. Cooking Steps" in raw_text or "4. Cooking Steps" in raw_text:
                    raw_text = "**Source:** AI Generated\n\n" + raw_text
                    break
            else:
                raw_text = f"Error: {response.status_code}, {response.text}"
                break

        results[dish_name] = clean_output(raw_text, format=format)

    return results[dish_names[0]] if single_input else results



def combine_models(model_1, model_2, image):
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Direct resize to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get YOLO detections
    yolo_result = get_yolo_result(model_1, image,output_dir="crop_output")

    # Prepare ResNet-50
    model_2.eval()
    model_2.to(device)

    # Store results
    cls_preds = []
    cls_confs = []

    # Process each crop like validation data
    for crop in yolo_result['food_images']:
        if crop.size == 0:
            continue  # Skip invalid detections

        # Convert to PIL RGB (matching ImageFolder's loading)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)

        # Apply validation transforms
        tensor = val_transform(crop_pil).unsqueeze(0).to(device)

        # Classify
        with torch.no_grad():
            outputs = model_2(tensor)

        # Get probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

        cls_preds.append(pred.item())
        cls_confs.append(conf.item())

    return  cls_preds, cls_confs

def get_category(category_file_path, class_to_idx_path):
    final_projection = {}
    with open(class_to_idx_path, 'r') as file:
        mod_dict = json.load(file)
    mod_dict = {value: key for key, value in mod_dict.items()}
    data_dict = {}
    with open(category_file_path, 'r', encoding='utf-8') as file:
        # Skip the header line
        next(file)
        for line in file:
            # Strip newline and split by tab
            parts = line.strip().split('\t')
            if len(parts) == 2:
                key, value = parts
                data_dict[key] = value
    for idx in mod_dict:
        final_projection[idx] = data_dict[str(mod_dict[idx])]
    return final_projection

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # MODEL 1
    model_1 = YOLO("best_model1.pt")
    # MODEL 2
    model_2 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model_2.fc.in_features
    model_2.fc = nn.Linear(num_ftrs, 256)
    model_2.load_state_dict(torch.load('food_recognition_model_resnet50_v3.pth',
                                       map_location=device))
    category_dict = get_category('category.txt', "class_to_idx.json")

    # MODEL 3
    path = kagglehub.dataset_download("irkaal/foodcom-recipes-and-reviews")
    recipes_df = pd.read_csv(os.path.join(path, "recipes.csv"))
    API_KEY = "sk-9618264c69e7485e9fe1d9ba92102614"

    # PROCESSING
    input_dir = "98_15345.jpg"

    image = cv2.imread(input_dir)

    predictions_idx, _ = combine_models(model_1, model_2, image)
    predictions_food = []

    for i in predictions_idx:
        predictions_food.append(category_dict[i])
    print(predictions_food)

    with open('recipes_output.md', 'w') as file:
        for i in predictions_food:
            markdown_text = generate_recipe(i, format="markdown")
            file.write(markdown_text + '\n\n')