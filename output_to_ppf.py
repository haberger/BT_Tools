
import open3d as o3d

xres = 1280
yres = 720
fx = 909.9260864257812
fy = 907.9168701171875
cx = 643.5625
cy = 349.0171813964844

data_path = "/home/david/UNI/BachelorThesis/implementation/Render/data/"

if __name__ == '__main__':

    for i in range(10, 64):
        pcd = o3d.io.read_point_cloud("output-point-cloud/0000000" + str(i) + '-output-pointcloud.ply')
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        o3d.io.write_point_cloud("pcd/manipulation-scene{}.pcd".format(i + 1), pcd)

        a_file = open("pcd/manipulation-scene{}.pcd".format(i+1), "rb")

        list_of_lines = a_file.readlines()

        print(list_of_lines[6])

        list_of_lines[6] = b'WIDTH 640\n'
        list_of_lines[7] = b'HEIGHT 360\n'

        a_file = open("pcd/manipulation-scene{}.pcd".format(i+1), "wb")

        a_file.writelines(list_of_lines)

        a_file.close()