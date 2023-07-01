import cv2
import math as m
import mediapipe as mp


def dot_prod(a, b, c):
    return (a[0]-b[0])*(c[0]-b[0]) + (a[1]-b[1])*(c[1]-b[1])

def sq_dist_btwn(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

def angle_btwn1(a, b, c):
    return int(m.acos(((a[0]-b[0])*(c[0]-b[0]) + (a[1]-b[1])*(c[1]-b[1])) / m.sqrt(((a[0]-b[0])**2 + (a[1]-b[1])**2)*((c[0]-b[0])**2 + (c[1]-b[1])**2))) * 180 / m.pi)

def angle_btwn2(l, n, r):
    dp = dot_prod(l, n, r)
    ln = sq_dist_btwn(l, n)
    rn = sq_dist_btwn(r, n)
    return int(m.acos(dp / m.sqrt(ln * rn)) * 180 / m.pi)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_default_pose = mp_drawing_styles.get_default_pose_landmarks_style()
mp_con = mp_pose.POSE_CONNECTIONS
lm_name = mp_pose.PoseLandmark

pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

w = 600
h = 400
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, img = cap.read()
    # img = cv2.resize(img, (w, h))

    results = pose.process(img)
    lm = results.pose_landmarks
    if lm:
        # mp_drawing.draw_landmarks(img, lm, mp_con, landmark_drawing_spec=mp_default_pose)
        n = (lm.landmark[lm_name.NOSE].x * w, lm.landmark[lm_name.NOSE].y * h)
        ls = (lm.landmark[lm_name.LEFT_SHOULDER].x * w, lm.landmark[lm_name.LEFT_SHOULDER].y * h)
        rs = (lm.landmark[lm_name.RIGHT_SHOULDER].x * w, lm.landmark[lm_name.RIGHT_SHOULDER].y * h)
        if angle_btwn1(ls, n, rs) < 30:
            lh = (lm.landmark[lm_name.LEFT_HIP].x * w, lm.landmark[lm_name.LEFT_HIP].y * h)
            rh = (lm.landmark[lm_name.RIGHT_HIP].x * w, lm.landmark[lm_name.RIGHT_HIP].y * h)
            lv = (lh[0], ls[1])
            rv = (rh[0], rs[1])
            avg_posture_angle = (angle_btwn1(lv, lh, ls) + angle_btwn1(rv, rh, rs)) / 2
            if avg_posture_angle < 6:
                cv2.putText(img, 'good posture, angle: ' + str(avg_posture_angle), (20, 20), font, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(img, 'bad posture, angle: ' + str(avg_posture_angle), (20, 20), font, 0.9, (0, 0, 255), 2)
        else:
            cv2.putText(img, 'not facing sideways', (20, 20), font, 0.9, (255, 0, 0), 2)
        
        # print(l[0], l[1])
    cv2.imshow('Possy', img)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()