from flask import Flask, render_template, request, send_file, redirect, session, jsonify
from werkzeug.utils import secure_filename
from hurry.filesize import size
from datetime import datetime
from flask_fontawesome import FontAwesome
from flask_qrcode import QRcode
from pathlib import Path
import os
import mimetypes
import sys
import re
import json
import zipfile
import filetype
from CLIP.deploy_main import main

from urllib.parse import unquote
import socket

hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)
print("Your Computer Name is: " + hostname)
print("Your Computer IP Address is: " + IPAddr)



app = Flask(__name__)
#app.config["SERVER_NAME"] = "wifile.com"
app.secret_key = 'my_secret_key'

# FoNT AWESOME
fa = FontAwesome(app)
# QRcode
qrcode = QRcode(app)
# Config file
config = os.path.abspath(os.path.join(os.path.dirname(__file__), "config.json"))
with open(config) as json_data_file:
    config_data = json.load(json_data_file)
hiddenList = config_data["Hidden"]
favList = config_data["Favorites"]
password = config_data["Password"]
maxFileNameLength = config_data["maxFileNameLength"]
sort_by_selected = config_data["sort_by_selected"]

sorted_label_dict = {0:"Alphabetical", 1:"Date Created", 2:"Date Modified", 3:"Size"}
sorted_label = sorted_label_dict[sort_by_selected]

currentDirectory = config_data["rootDir"]
osWindows = False  # Not Windows
default_view = 0
tp_dict = {'image': [['png', "jpg", 'svg'], 'image-icon.png'],
           'audio': [['mp3', 'wav'], 'audio-icon.png'], 
           'video': [['mp4', 'flv'], 'video-icon.png'],
           "pdf": [['pdf'], 'pdf-icon.png'],
           "word": [['docx', 'doc'], 'doc-icon.png'],
           "txt": [['txt'], 'txt-icon.png'],
           "compressed":[["zip", "rar"], 'copressed-icon.png'],
           "code": [['css', 'scss', 'html', 'py', 'js', 'cpp'], 'code-icon.png']
           }
supported_formats = video_types = ['mp4', "webm", "opgg",'mp3', 'pdf', 'txt', 'html', 'css', 'svg', 'js', 'png', 'jpg']

if 'win32' in sys.platform or 'win64' in sys.platform:
    # import win32api
    osWindows = True
    # WINDOWS FEATURE
    # drives = win32api.GetLogicalDriveStrings()
    # drives=drives.replace('\\','')
    # drives = drives.split('\000')[:-1]
    # drives.extend(favList)
    # favList=drives

if(len(favList) > 3):
    favList = favList[0:3]
# print(favList)
# if(len(favList)>0):
#     for i in range(0,len(favList)):

#         favList[i]=favList[i].replace('\\','>') #CHANGE FOR MAC

# WINDOWS FEATURE
# drives = win32api.GetLogicalDriveStrings()
# drives=drives.replace('\\','')
# drives = drives.split('\000')[:-1]
# drives.extend(favList)
# favList=drives

def make_zipfile(output_filename, source_dir):
    relroot = os.path.abspath(os.path.join(source_dir, os.pardir))
    with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zip:
        for root, dirs, files in os.walk(source_dir):
            # add directory (needed for empty dirs)
            zip.write(root, os.path.relpath(root, relroot))
            for file in files:
                filename = os.path.join(root, file)
                if os.path.isfile(filename):  # regular files only
                    arcname = os.path.join(
                        os.path.relpath(root, relroot), file)
                    zip.write(filename, arcname)


@app.route('/login/')
@app.route('/login/<path:var>')
def loginMethod(var=""):
    global password
    # print("LOGGING IN")
    # print(var)
    if(password == ''):
        session['login'] = True
    if('login' in session):
        return redirect('/'+var)
    else:
        return render_template('login.html')


@app.route('/login/', methods=['POST'])
@app.route('/login/<path:var>', methods=['POST'])
def loginPost(var=""):
    global password
    text = request.form['text']
    if(text == password):
        session['login'] = True
        return redirect('/'+var)
    else:
        return redirect('/login/'+var)


@app.route('/logout/')
def logoutMethod():
    if('login' in session):
        session.pop('login', None)
    return redirect('/login/')

# @app.route('/exit/')
# def exitMethod():
#    exit()


def hidden(path):
    for i in hiddenList:
        if i != '' and i in path:
            return True
    return False


def changeDirectory(path):
    global currentDirectory, osWindows
    pathC = path.split('/')
    # print(path)
    if(osWindows):
        myPath = '//'.join(pathC)+'//'
    else:
        myPath = '/'+'/'.join(pathC)
    # print(myPath)
    myPath = unquote(myPath)
    # print("HELLO")
    # print(myPath)
    try:
        os.chdir(myPath)
        ans = True
        if (osWindows):
            if(currentDirectory.replace('/', '\\') not in os.getcwd()):
                ans = False
        else:
            if(currentDirectory not in os.getcwd()):
                ans = False
    except Exception as e:
        ans = False
    return ans

# def getDirList():
#     dList= list(filter(lambda x: os.path.isdir(x), os.listdir('.')))
#     finalList = []
#     curDir=os.getcwd()

#     for i in dList:
#         if(hidden(curDir+'/'+i)==False):
#             finalList.append(i)

#     return(finalList)

@app.route('/changeView')
def changeView():
    global default_view
    # print('view received')
    v = int(request.args.get('view', 0))
    if v in [0, 1]:
        default_view = v
    else:
        default_view = 0

    return jsonify({
        "txt": default_view,
    })

@app.route('/changeSort')
def toggleSort():
    global sort_by_selected
    # 0 == alphabetical
    # 1 == date created
    # 2 == date modified
    # 3 == size+alphabetical

    sort_by_selected = sort_by_selected + 1 if sort_by_selected < 3 else 0

    session['sorting_default'] = sort_by_selected


    return jsonify({
        "txt": sort_by_selected,
    })



def getDirList():
    # print(default_view)
    global maxFileNameLength, tp_dict, hostname
    dList = list(os.listdir('.'))
    dList = list(filter(lambda x: os.path.isdir(x), os.listdir('.')))
    dir_list_dict = []
    fList = list(filter(lambda x: not os.path.isdir(x), os.listdir('.')))
    file_list_dict = []
    curDir = os.getcwd()
    # print(os.stat(os.getcwd()))
    for i in dList:
        if(hidden(curDir+'/'+i) == False):
            image = 'folder5.png'
            if len(i) > maxFileNameLength:
                dots = "..."
            else:
                dots = ""
            dir_stats = os.stat(i)
            temp_dir = {}
            temp_dir['f'] = i[0:maxFileNameLength]+dots
            temp_dir['f_url'] = re.sub("#", "|HASHTAG|", i)
            temp_dir['currentDir'] = curDir
            temp_dir['f_complete'] = i
            temp_dir['image'] = image
            temp_dir['dtc'] = datetime.utcfromtimestamp(dir_stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
            temp_dir['dtm'] = datetime.utcfromtimestamp(dir_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            temp_dir['size'] = "---"

            dir_list_dict.append(temp_dir)



    from utils import get_file_extension
    for i in fList:
        if(hidden(curDir+'/'+i) == False):
            image = None
            try:
                tp = get_file_extension(i)
                for file_type in tp_dict.values():
                    if tp in file_type[0]:
                        image = "files_icon/"+file_type[1]
                        break
                tp = "" if not tp else tp
            except Exception as e:
                pass
            if not image:
                image = 'files_icon/unknown-icon.png'
            if len(i) > maxFileNameLength:
                dots = "..."
            else:
                dots = ""
            temp_file = {}
            temp_file['f'] = i[0:maxFileNameLength]+dots
            temp_file['f_url'] = re.sub("#", "|HASHTAG|", i)
            temp_file['currentDir'] = curDir
            temp_file['f_complete'] = i
            temp_file['image'] = image
            temp_file['supported'] = True if tp.lower() in supported_formats else False
            try:
                dir_stats = os.stat(i)
                temp_file['dtc'] = datetime.utcfromtimestamp(dir_stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
                temp_file['dtm'] = datetime.utcfromtimestamp(dir_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                temp_file['size'] = size(dir_stats.st_size)
                temp_file['size_b'] = dir_stats.st_size
            except Exception as e:
                temp_file['dtc'] = "---"
                temp_file['dtm'] = "---"
                temp_file['size'] = "---"
                temp_file['size_b'] = -1

            file_list_dict.append(temp_file)

    
    return sort_structure(dir_list_dict, file_list_dict)


def sort_structure(dir_list_dict, file_list_dict):

    global sort_by_selected, sorted_label_dict

    sort_by_selected = session.get('sorting_default', sort_by_selected)

    if sort_by_selected == 0:
        dir_list_dict = sorted(dir_list_dict, key=lambda x: x['f'].lower())
        file_list_dict = sorted(file_list_dict, key=lambda x: x['f'].lower())
        
    elif sort_by_selected == 1:
        dir_list_dict = sorted(dir_list_dict, key=lambda x: x['dtc'])
        file_list_dict = sorted(file_list_dict, key=lambda x: x['dtc'])
        
    elif sort_by_selected == 2:
        dir_list_dict = sorted(dir_list_dict, key=lambda x: x['dtm'])
        file_list_dict = sorted(file_list_dict, key=lambda x: x['dtm'])
        
    elif sort_by_selected == 3:
        dir_list_dict = sorted(dir_list_dict, key=lambda x: x['f'].lower())
        file_list_dict = sorted(file_list_dict, key=lambda x: x['size_b'])
        

    sorted_by_label = sorted_label_dict[sort_by_selected]

    return dir_list_dict, file_list_dict, sorted_by_label

def getFileList():
    dList = list(filter(lambda x: os.path.isfile(x), os.listdir('.')))
    finalList = []
    curDir = os.getcwd()
    for i in dList:
        if(hidden(curDir+'/'+i) == False):
            finalList.append(i)
    return(finalList)


@app.route('/files/', methods=['GET'])
@app.route('/files/<path:var>', methods=['GET'])
def filePage(var=""):
    global default_view
    if('login' not in session):
        return redirect('/login/files/'+var)
    # print(var)
    if(changeDirectory(var) == False):
        # Invalid Directory
        print("Directory Doesn't Exist")
        return render_template('404.html', errorCode=300, errorText='Invalid Directory Path', favList=favList)

    try:
        dir_dict, file_dict, sorted_label_current = getDirList()
        if default_view == 0:
            var1, var2 = "DISABLED", ""
            default_view_css_1, default_view_css_2 = '', 'style=display:none'
        else:
            var1, var2 = "", "DISABLED"
            default_view_css_1, default_view_css_2 = 'style=display:none', ''
    except Exception as e:
        return render_template('404.html', errorCode=200, errorText='Permission Denied {}'.format(e), favList=favList)
    if osWindows:
        cList = var.split('/')
        var_path = '<a style = "color:black;"href = "/files/' + \
            cList[0]+'">'+unquote(cList[0])+'</a>'
        for c in range(1, len(cList)):
            var_path += ' / <a style = "color:black;"href = "/files/' + \
                '/'.join(cList[0:c+1])+'">'+unquote(cList[c])+'</a>'
    else:
        cList = var.split('/')
        var_path = '<a href = "/files/"><img src = "/static/root.png" style = "height:25px;width: 25px;">&nbsp;</a>'
        for c in range(0, len(cList)):
            var_path += ' / <a style = "color:black;"href = "/files/' + \
                '/'.join(cList[0:c+1])+'">'+unquote(cList[c])+'</a>'
    return render_template('home.html', currentDir=var, favList=favList, default_view_css_1=default_view_css_1, default_view_css_2=default_view_css_2, view0_button=var1, view1_button=var2, currentDir_path=var_path, dir_dict=dir_dict, file_dict=file_dict, sorted_label_current=sorted_label_current)

@app.route('/get_drives', methods=['GET'])
def get_drives():
    if osWindows:
        drives = [f"{d}:\\" for d in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if os.path.exists(f"{d}:\\")]
    else:
        drives = ["/"]  # Đối với hệ điều hành không phải Windows
    return jsonify(drives)

import shutil

@app.route('/', methods=['GET'])
def homePage():
    global currentDirectory, osWindows
    if('login' not in session):
        return redirect('/login/')
    if osWindows:
        if(currentDirectory == ""):
            return redirect('/files/C:')
        else:
            # cura = currentDirectory
            cura = '>'.join(currentDirectory.split('\\'))
            return redirect('/files/'+cura)
    else:
        return redirect('/files/'+currentDirectory)
        # REDIRECT TO UNTITLED OR C DRIVE FOR WINDOWS OR / FOR MAC


# @app.route('/process_folder', methods=['POST'])
# def process_folder():
#     files = request.files.getlist('files[]')
#     folder_path = request.form.get('folder_path')  # Lấy đường dẫn thư mục từ form data
#     upload_folder = os.path.join(app.root_path, 'static/test/images')
#     os.makedirs(upload_folder, exist_ok=True)
#
#     # Kiểm tra xem folder_path có tồn tại không
#     if not folder_path:
#         return jsonify({"error": "Không tìm thấy đường dẫn thư mục."}), 400
#
#     # Gọi hàm main và lấy danh sách ảnh
#     query = "Một người phụ nữ chơi tennis mặc áo đỏ."
#     full_folder_path = os.path.join(app.root_path, folder_path)
#     list_images = main(query, image_path=full_folder_path)  # Truyền đường dẫn thư mục đầy đủ vào `main`
#
#     # Sao chép ảnh từ output của `main` vào thư mục tĩnh
#     selected_images = []
#     for group in list_images:
#         for image_path in group:
#             # Chuyển đường dẫn thành đường dẫn hệ thống
#             src_path = image_path.replace("\\", "/")
#             filename = os.path.basename(src_path)
#             dst_path = os.path.join(upload_folder, filename)
#
#             # Copy file nếu chưa tồn tại
#             if not os.path.exists(dst_path):
#                 shutil.copy(src_path, dst_path)
#
#             # Tạo URL tĩnh
#             selected_images.append(f'/static/test/images/{filename}')
#
#     return jsonify({"images": selected_images})
@app.route('/query_image')
def query_image():
    return render_template('query_image.html')

@app.route('/send-text', methods=['POST'])
def send_text():
    data = request.json
    text = data.get('text', '')

    if text:
        print(f"Nội dung nhận được: {text}")
        return jsonify({"message": f"Nội dung đã nhận: {text}"})
    else:
        return jsonify({"error": "Không có nội dung nào được gửi"}), 400


@app.route('/upload-folder-path', methods=['POST'])
def upload_folder_path():
    data = request.json
    folder_path = data.get('folderPath', '')

    if folder_path:
        # Xử lý thông tin đường dẫn thư mục (nếu cần)
        print(f"Đường dẫn thư mục nhận được: {folder_path}")
        return jsonify({"message": f"Đã nhận thư mục: {folder_path}"})
    else:
        return jsonify({"error": "Không nhận được thông tin thư mục"}), 400


@app.route('/browse/<path:var>', defaults={"browse":True})
@app.route('/download/<path:var>', defaults={"browse":False})
def browseFile(var, browse):
    var = var.replace("|HASHTAG|", "#")
    if('login' not in session):
        return redirect('/login/download/'+var)
    # os.chdir(currentDirectory)
    pathC = unquote(var).split('/')
    #print(var)
    if(pathC[0] == ''):
        pathC.remove(pathC[0])
    # if osWindows:
    #     fPath = currentDirectory+'//'.join(pathC)
    # else:
    #     fPath = '/'+currentDirectory+'//'.join(pathC)
    if osWindows:
        fPath = '//'.join(pathC)
    else:
        fPath = '/'+'//'.join(pathC)
    # print("HELLO")
    # print('//'.join(fPath.split("//")[0:-1]))
    # print(hidden('//'.join(fPath.split("//")[0:-1])))
    f_path_hidden = '//'.join(fPath.split("//")[0:-1])
    if(hidden(f_path_hidden) == True or changeDirectory(f_path_hidden) == False):
        # FILE HIDDEN
        return render_template('404.html', errorCode=100, errorText='File Hidden', favList=favList)
    fName = pathC[len(pathC)-1]
    #print(fPath)
    if browse:
        from utils import is_media
        is_media_file = is_media(fPath)
        if is_media_file:
            from utils import get_file
            return get_file(fPath, is_media_file)
    return send_file(fPath)
    try:
        return send_file(fPath, download_name=fName)
    except Exception as e:
        return render_template('404.html', errorCode=200, errorText='Permission Denied {}'.format(e), favList=favList)


@app.route('/downloadFolder/<path:var>')
def downloadFolder(var):
    if('login' not in session):
        return redirect('/login/downloadFolder/'+var)
    pathC = var.split('/')
    if(pathC[0] == ''):
        pathC.remove(pathC[0])
    if osWindows:
        fPath = '//'.join(pathC)
    else:
        fPath = '/'+'//'.join(pathC)
    f_path_hidden = '//'.join(fPath.split("//")[0:-1])
    if(hidden(f_path_hidden) == True or changeDirectory(f_path_hidden) == False):
        # FILE HIDDEN
        return render_template('404.html', errorCode=100, errorText='File Hidden', favList=favList)
    fName = pathC[len(pathC)-1]+'.zip'
    downloads_folder = str(Path.home() / "Downloads\\temp")
    if not os.path.exists(downloads_folder):
        os.mkdir(downloads_folder)
    try:
        make_zipfile(downloads_folder+'\\abc.zip', os.getcwd())
        return send_file(downloads_folder+'\\abc.zip', attachment_filename=fName)
    except Exception as e:
        print(e)
        return render_template('404.html', errorCode=200, errorText='Permission Denied {}'.format(e), favList=favList)


@app.errorhandler(404)
def page_not_found(e):
    if('login' not in session):
        return redirect('/login/')
    # note that we set the 404 status explicitly
    return render_template('404.html', errorCode=404, errorText='Page Not Found', favList=favList), 404


@app.route('/upload/', methods=['GET', 'POST'])
@app.route('/upload/<path:var>', methods=['GET', 'POST'])
def uploadFile(var=""):
    if('login' not in session):
        return render_template('login.html')
    text = ""
    if request.method == 'POST':
        pathC = var.split('/')
        if(pathC[0] == ''):
            pathC.remove(pathC[0])
        # if osWindows:
        #     fPath = currentDirectory+'//'.join(pathC)
        # else:
        #     fPath = '/'+currentDirectory+'//'.join(pathC)
        if osWindows:
            fPath = '//'.join(pathC)
        else:
            fPath = '/'+'//'.join(pathC)
        f_path_hidden = fPath
        # print(f_path_hidden)
        # print(hidden(f_path_hidden))
        if(hidden(f_path_hidden) == True or changeDirectory(f_path_hidden) == False):
            # FILE HIDDEN
            return render_template('404.html', errorCode=100, errorText='File Hidden', favList=favList)
        files = request.files.getlist('files[]')
        fileNo = 0
        for file in files:
            file.filename = secure_filename(file.filename) # ensure file name is secure
            fupload = os.path.join(fPath, file.filename)
            if not os.path.exists(fupload):
                try:
                    file.save(fupload)
                    print(file.filename + ' Uploaded')
                    text = text + file.filename + ' Uploaded<br>'

                    fileNo = fileNo + 1
                except Exception as e:
                    print(file.filename + ' Failed with Exception '+str(e))
                    text = text + file.filename + \
                        ' Failed with Exception '+str(e) + '<br>'
                    continue
            else:
                print(file.filename +
                      ' Failed because File Already Exists or File Type Issue')
                text = text + file.filename + \
                    ' Failed because File Already Exists or File Type not secure <br>'
    fileNo2 = len(files)-fileNo
    return render_template('uploadsuccess.html', text=text, fileNo=fileNo, fileNo2=fileNo2, favList=favList)


@app.route('/qr/<path:var>')
def qrFile(var):
    global hostname
    if('login' not in session):
        return redirect('/login/qr/'+var)
    # os.chdir(currentDirectory)
    pathC = unquote(var).split('/')
    if(pathC[0] == ''):
        pathC.remove(pathC[0])
    if osWindows:
        fPath = '//'.join(pathC)
    else:
        fPath = '/'+'//'.join(pathC)
    f_path_hidden = '//'.join(fPath.split("//")[0:-1])
    if(hidden(f_path_hidden) == True or changeDirectory(f_path_hidden) == False):
        # FILE HIDDEN
        return render_template('404.html', errorCode=100, errorText='File Hidden', favList=favList)
    fName = pathC[len(pathC)-1]
    qr_text = 'http://'+hostname+"//download//"+fPath
    return send_file(qrcode(qr_text, mode="raw"), mimetype="image/png")
    return send_file(fPath, attachment_filename=fName)


if __name__ == '__main__':
    local = "127.0.0.1"
    public = '0.0.0.0'
    app.run(host=public, debug=True, port=80)