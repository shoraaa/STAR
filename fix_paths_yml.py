import os
import re

def fix_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.yml') or file.endswith('.yaml') or file.endswith('.json'):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                modified = False
                
                # Replace absolute path with relative path
                if '/home/shora/Research/NRS_Survey/NRS/' in content:
                    # just change it to relative since it's loaded from the same directory usually
                    # wait, the paths in config.yml usually point to weights inside the same directory or child directory.
                    # let's just make it relative to the yml file.
                    # /home/shora/Research/NRS_Survey/NRS/Construction/single-stage/appending/9_ReLD/CVRP/weights/ReLD/model_epoch_90.pt
                    # becomes weights/ReLD/model_epoch_90.pt
                    new_content = re.sub(r'/home/shora/Research/NRS_Survey/[a-zA-Z0-9_/-]+/weights/', 'weights/', content)
                    if new_content != content:
                        modified = True
                        content = new_content
                        
                if modified:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"Fixed {path}")

fix_directory('survey/NRS')
