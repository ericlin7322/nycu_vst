import os
from flask import render_template, Flask, send_from_directory, request

app = Flask(__name__)


#@app.after_request
#def add_header(response):
#    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
#    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
#    response.headers['Pragma'] = 'no-cache'
#    response.headers['Expires'] = '0'
#    return response


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_location', methods=['POST'])
def receive_location():
    data = request.get_json()
    x = int(data['tempx'])
    y = int(data['tempy'])

    # Process the location data as needed
    print(f"Received location - x: {x}, y: {y}")
    loaded_bboxes = []
    existing_ids = set()

    # Open the file in read mode
    with open('bounding_boxes.txt', 'r') as file:
            # Iterate through each line in the file
            for line in file:
            # Split the line into parts based on the comma
                parts = line.split(',')
                
                # Extract tracking ID and bounding box from the parts
                tracking_id = int(parts[0].strip())
                
                # Extract the bounding box as a string and convert it to a tuple
                x1 = int(parts[1].strip())
                y1 = int(parts[2].strip())
                x2 = int(parts[3].strip())
                y2 = int(parts[4].strip())
                
                width = int(parts[5].strip())
                height = int(parts[6].strip())
                
                x1 = int(x1 * 100 / width)
                x2 = int(x2 * 100 / width)
                y1 = int(y1 * 100 / height)
                y2 = int(y2 * 100 / height)
                
                # print(x, y, x1, x2, y1, y2)
                
                # Create a dictionary with tracking ID and bounding box
                bounding_box_info = {'tracking_id': tracking_id, 'bbox': (x1,x2,y1,y2)}
                
                loaded_bboxes.append(bounding_box_info)

                
                if x1 <= x and x <= x2 and y1 <= y and y <= y2:
                    # print(tracking_id)
                    exist = False
                    
                    with open('tracking_ids.txt', 'r') as file1:
                        for l in file1:
                            print(l)
                            existing_ids.add(int(l.strip()))
                    
                    
                    for temp in existing_ids:
                        if temp == tracking_id:
                            existing_ids.remove(tracking_id)
                            exist = True
                            break
                        
                    if not exist:
                        existing_ids.add(tracking_id)
                    
                    with open('tracking_ids.txt', 'w') as output_file:
                        for temp in existing_ids:
                            print(temp)
                            output_file.write(f'{temp}\n')
                            
    return 'Location received successfully!'

@app.route('/video/<string:file_name>')
def stream(file_name):
    video_dir = './video'
    return send_from_directory(video_dir, file_name)


if __name__ == '__main__':
    app.run()
