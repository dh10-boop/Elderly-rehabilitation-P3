import math
import cv2
import mediapipe as mp


def calculate_angle(a, b, c):
    """计算以 b 为顶点的二维夹角，返回 0～180 度。"""
    direction1 = math.atan2(a[1] - b[1], a[0] - b[0])
    direction2 = math.atan2(c[1] - b[1], c[0] - b[0])

    angle = abs(math.degrees(direction2 - direction1))

    if angle > 180:
        angle = 360 - angle

    return angle


def get_point(landmark_list, point_id, width, height):
    """按关键点名称从列表取点，再转换成画面像素坐标。"""
    index = point_id.value
    landmark = landmark_list[index]

    x = int(landmark.x * width)
    y = int(landmark.y * height)

    return (x, y)


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
P = mp_pose.PoseLandmark

pose = mp_pose.Pose(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    pose.close()
    raise SystemExit("摄像头打开失败")

joint_list = [
    ("LeftShoulder", P.LEFT_ELBOW, P.LEFT_SHOULDER, P.LEFT_HIP),
    ("RightShoulder", P.RIGHT_ELBOW, P.RIGHT_SHOULDER, P.RIGHT_HIP),
    ("LeftElbow", P.LEFT_SHOULDER, P.LEFT_ELBOW, P.LEFT_WRIST),
    ("RightElbow", P.RIGHT_SHOULDER, P.RIGHT_ELBOW, P.RIGHT_WRIST),
    ("LeftHip", P.LEFT_SHOULDER, P.LEFT_HIP, P.LEFT_KNEE),
    ("RightHip", P.RIGHT_SHOULDER, P.RIGHT_HIP, P.RIGHT_KNEE),
    ("LeftKnee", P.LEFT_HIP, P.LEFT_KNEE, P.LEFT_ANKLE),
    ("RightKnee", P.RIGHT_HIP, P.RIGHT_KNEE, P.RIGHT_ANKLE),
]

while True:
    ret, frame = cap.read()

    if not ret:
        print("读取画面失败")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
        )

        landmark_list = results.pose_landmarks.landmark
        height, width, _ = frame.shape

        for name, a_id, b_id, c_id in joint_list:
            a = get_point(landmark_list, a_id, width, height)
            b = get_point(landmark_list, b_id, width, height)
            c = get_point(landmark_list, c_id, width, height)

            angle = calculate_angle(a, b, c)

            cv2.putText(
                frame,
                f"{name}:{angle:.0f}",
                b,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

    cv2.imshow("Pose Angles", frame)

    if cv2.waitKey(1) == ord("q"):
        break

pose.close()
cap.release()
cv2.destroyAllWindows()
