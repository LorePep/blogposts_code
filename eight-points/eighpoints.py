import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform.rotation import Rotation as R

COLOR_PALETTE = np.random.randint(0, 255, (100, 3)) / 255


def main():
    camera_1, camera_2, T12 = _generate_camera_frames(0, 0.3, 0.2, -1.5)
    points = _generate_3d_points_in_world(num_points=8)

    _plot_3d_world(camera_1, camera_2, points)

    points_1, points_2 = _transform_points_in_cameras(
        points, camera1=camera_1, camera2=camera_2
    )

    _plot_camera_images(points_1, points_2)

    chi = _compute_kronecker(points_1, points_2)

    U, _, V = _estimate_decomponsed_essential_matrix(chi)

    possible_r_t = _extract_rot_transl(U, V)
    print(possible_r_t[0][0])
    print(possible_r_t[2][0])
    print(T12)


def _generate_3d_points_in_world(num_points):
    xs = np.random.uniform(low=5, high=10, size=num_points)
    ys = np.random.uniform(low=0, high=5, size=num_points)
    zs = np.random.uniform(low=5, high=10, size=num_points)
    h = np.ones((num_points,))

    return np.array((xs, ys, zs, h))


def _get_rotation_matrix(r, p, y):
    return R.from_euler("zyx", [r, p, y]).as_matrix()


def _plot_camera_frame(ax, camera, scale=1):
    center = camera[:3, 3]
    x = scale * camera[:3, 0]
    y = scale * camera[:3, 1]
    z = scale * camera[:3, 2]

    ax.plot(
        (center[0], center[0] + x[0]),
        (center[1], center[1] + x[1]),
        (center[2], center[2] + x[2]),
        "r",
    )
    ax.plot(
        (center[0], center[0] + y[0]),
        (center[1], center[1] + y[1]),
        (center[2], center[2] + y[2]),
        "g",
    )
    ax.plot(
        (center[0], center[0] + z[0]),
        (center[1], center[1] + z[1]),
        (center[2], center[2] + z[2]),
        "b",
    )

    return ax


def _generate_camera_frames(r, p, y, t_x):
    camera1 = np.array(
        ((-1, 0, 0, 11.0), (0, 0, 1, -10.0), (0, 1, 0, 4.0), (0, 0, 0, 1),)
    )

    rot = _get_rotation_matrix(r, p, y)
    T12 = np.eye(4)
    T12[:3, :3] = rot
    T12[0, 3] = t_x
    camera2 = np.dot(camera1, T12)

    return camera1, camera2, T12


def _transform_points_in_cameras(points, camera1, camera2):
    points_in_cam_1 = np.dot(np.linalg.inv(camera1), points)
    points_in_cam_2 = np.dot(np.linalg.inv(camera2), points)

    for i in range(points.shape[1]):
        points_in_cam_1[:2, i] /= points_in_cam_1[2, i]
        points_in_cam_1[2, i] = 1
        points_in_cam_2[:2, i] /= points_in_cam_2[2, i]
        points_in_cam_2[2, i] = 1

    return points_in_cam_1[:3, :], points_in_cam_2[:3, :]


def _plot_3d_world(camera_1_pose, camera_2_pose, points):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    for i in range(points.shape[1]):
        ax.scatter(points[0, i], points[1, i], points[2, i], color=COLOR_PALETTE[i])
    ax = _plot_camera_frame(ax, camera_1_pose)
    ax = _plot_camera_frame(ax, camera_2_pose)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_xlim([0, 20])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-2, 18])
    plt.title("3D worlds camera frames and 3D points")

    plt.savefig("3d_world.png")


def _plot_camera_images(points_1, points_2):
    _, axs = plt.subplots(1, 2)
    for i in range(points_1.shape[1]):
        axs[0].scatter(-points_1[0, i], points_1[1, i], color=COLOR_PALETTE[i])
    for i in range(points_1.shape[1]):
        axs[1].scatter(-points_2[0, i], points_2[1, i], color=COLOR_PALETTE[i])
    axs[0].set_xlim([-1, 1])
    axs[0].set_ylim([-1, 1])
    axs[0].set_title("Image from camera 1")
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].set_title("Image from camera 2")
    axs[1].set_xlim([-1, 1])
    axs[1].set_ylim([-1, 1])
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    plt.savefig("images.png")


def _compute_kronecker(points_1, points_2):
    x1 = points_1[0, :]
    y1 = points_1[1, :]
    x2 = points_2[0, :]
    y2 = points_2[1, :]

    chi = np.ones((8, 9))
    chi[:, 0] = np.multiply(x2, x1)
    chi[:, 1] = np.multiply(x2, y1)
    chi[:, 2] = x2
    chi[:, 3] = np.multiply(y2, x1)
    chi[:, 4] = np.multiply(y2, y1)
    chi[:, 5] = y2
    chi[:, 6] = x1
    chi[:, 7] = y1

    return chi


def _estimate_decomponsed_essential_matrix(chi):
    _, _, V1 = np.linalg.svd(chi)
    F = V1[8, :].reshape(3, 3).T
    U, _, V = np.linalg.svd(F)
    if np.linalg.det(np.dot(U, V)) < 1:
        V = -V

    return U, np.diag((1, 1, 0)), V


def _extract_rot_transl(U, V):
    W = np.array(([0, -1, 0], [1, 0, 0], [0, 0, 1]))
    return [
        [np.dot(U, np.dot(W, V)), U[-1, :]],
        [np.dot(U, np.dot(W, V)), -U[-1, :]],
        [np.dot(U, np.dot(W.T, V)), U[-1, :]],
        [np.dot(U, np.dot(W.T, V)), -U[-1, :]],
    ]


if __name__ == "__main__":
    main()
