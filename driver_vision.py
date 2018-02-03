from networktables import NetworkTables
import cscore
import time

# tuples are (camera_name, camera_device_id)
cameras = [
    ('Camera 1', 0),
    ('Camera 2', 1),
]

def main():
    cs_instance = cscore.CameraServer.getInstance()
    table = NetworkTables.getTable("Preferences")

    res_w = int(table.getNumber('Camera Res Width', 320))
    res_h = int(table.getNumber('Camera Res Height', 200))
    fps = int(table.getNumber('Camera FPS', 30))

    camera_objects = []

    for cam_idx, cam_config in enumerate(cameras):
        name, dev_id = cam_config

        camera_obj = cscore.UsbCamera(name=name, dev=dev_id)
        camera_obj.setResolution(res_w, res_h)
        camera_obj.setFPS(fps)

        camera_objects.append(camera_obj)
        #camera_chooser.addDefault(name, cam_idx)

    cam_server = cs_instance.addServer(name='camera_server')
    current_selected = int(table.getNumber('Selected Camera', 0))

    if current_selected >= len(camera_objects):
        current_selected = 0


    cam_server.setSource(camera_objects[current_selected])

    while True:
        selected_camera = int(table.getNumber('Selected Camera', 0))

        if (
            selected_camera < len(camera_objects)
            and selected_camera != current_selected
        ):
            res_w = int(table.getNumber('Camera Res Width', 320))
            res_h = int(table.getNumber('Camera Res Height', 200))
            fps = int(table.getNumber('Camera FPS', 30))

            camera_obj = camera_objects[selected_camera]

            camera_obj.setResolution(res_w, res_h)
            camera_obj.setFPS(fps)

            cam_server.setSource(camera_obj)
            current_selected = selected_camera

        time.sleep(0.02)
