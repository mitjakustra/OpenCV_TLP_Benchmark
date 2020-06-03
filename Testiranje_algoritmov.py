import numpy as np
import cv2 as cv
import sys
import time
import keyboard

'''FUNKCIONALNOST PROGRAMA:
    -program najprej zahteva ime sekvence kot vhodni podatek
    -program uvozi izbrano sekvenco in njeno pripadajoco groundtruth_rect.txt datoteko
    -program odpre se izhodno (TRACKERTYPE)_test_result.txt datoteko
    -program postavi zacetni pravokotnik na enako mesto kot je to storjeno v prvi
     vrstici groundtruth datoteke
    -program inicializira izbrani tip sledilnega algoritma in prvi pravokotnik ("bounding box")
    -program v neskoncni zanki oz. dokler mu ne zmanjka slicic iz sekvence isce
     nove pravokotnike in z njimi posodablja stare ter jih izrisuje prav tako kot
     tudi referencne iz groundtruth datoteke
    -program v izhodno .txt datoteko zapise v svojo vrstico za vsako slicico podatke v naslednjem formatu:
            FRAME_NUM, X_COR, Y_COR, WIDTH, HEIGHT, LOST
    -na zivi sliki sledenja se v zgornjem levem kotu izpisuje ime sledilnega alg., stevilo
     preostalih slicic do konca sekvence in hitrost algoritma v slicicah na sekundo ("FPS rate")
'''

frame_Num = 600
tr_list = ["BOOSTING", "MIL", "MEDIANFLOW", "TLD", "KCF", "MOSSE", "CSRT"]

print("Izberite zaporedno stevilko sekvence:")
print("1 - Alladin")
print("2 - Aquarium2")
print("3 - Badminton1")
print("4 - CarChase3")
print("5 - DriftCar1")
print("6 - ISS")
print("7 - Jet4")
print("8 - KinBall2")
print("9 - PolarBear1\n")

seq_name = input("Vnesite stevilko izbrane sekvence:")

if seq_name == '1':
    seq_name = "Alladin"
    print("Izbrana je sekvenca: Alladin")
elif seq_name == '2':
    seq_name = "Aquarium2"
    print("Izbrana je sekvenca: Aquarium2")
elif seq_name == '3':
    seq_name = "Badminton1"
    print("Izbrana je sekvenca: Badminton1")
elif seq_name == '4':
    seq_name = "CarChase3"
    print("Izbrana je sekvenca: CarChase3")
elif seq_name == '5':
    seq_name = "DriftCar1"
    print("Izbrana je sekvenca: DriftCar1")
elif seq_name == '6':
    seq_name = "ISS"
    print("Izbrana je sekvenca: ISS")
elif seq_name == '7':
    seq_name = "Jet4"
    print("Izbrana je sekvenca: Jet4")
elif seq_name == '8':
    seq_name = "KinBall2"
    print("Izbrana je sekvenca: KinBall2")
elif seq_name == '9':
    seq_name = "PolarBear1"
    print("Izbrana je sekvenca: PolarBear1")

for i in tr_list:
    tr_input_type = i

    path_result = "./data/" + seq_name + "/" + tr_input_type + "_test_result.txt"
    path_source = "./data/" + seq_name + "/img/%5d.jpg"
    path_gt = "./data/" + seq_name + "/groundtruth_rect.txt"

    #****** Odpiranje Ground_truth in branje prve vrstice *******************
    gt = open(path_gt, 'r')
    line1 = gt.readline().split(',')
    bbox0 = (int(line1[1]), int(line1[2]), int(line1[3]), int(line1[4]))

    # ***** Kreiranje datoteke za zapis rezultatov **************************
    test_result = open(path_result, "w")

    # ***** Uvoz sekvence slik *********************************************
    seq = cv.VideoCapture(path_source)

    # ***** Izhod ce sekvenca ni uvozena ***********************************
    if not seq.isOpened():
        print("Could not open video")
        sys.exit()

    frameCounter = 0

    # ***** Preberemo prvo sliko sekvence ***********************************
    ok, frame = seq.read()
    if not ok:
        print("Cannot read video file")
        sys.exit()

    frameCounter += 1

    # ***** Vnos stevila in tipov sledilcev********************************
    # to idejo zaenkrat opustimo, ker TLP primerja performance samo single-object tracking-a
    # trackerNo = int(input("Vnesi zeljeno stevilo trackerjev"))
    trackerNo = 1

    # ****** Vnos tipov sledilcev ******************************************
    trackerArray = []
    for i in range(trackerNo):
        #trackerType = str(input("Vnesi tip sledilca %d" % (i + 1)))
        trackerType = str(tr_input_type)
        trackerArray.append(trackerType)

    trackerArray = np.array(trackerArray)

    # ***** Funkcija za inicializacijo sledilca ****************************
    def trackerCreator(TrackerName):
        ''' TrackerName - (str) ime tipa sledilnega algoritma '''

        tracker_type = TrackerName
        if tracker_type == 'BOOSTING':
            tracker = cv.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv.TrackerCSRT_create()

        return tracker


    # ***** Funkcija za izbiro "bounding box-a" ***************************
    def boundingBoxSelect():
        cv.namedWindow('Select Bounding Box', cv.WINDOW_NORMAL)
        cv.resizeWindow('Select Bounding Box', frame.shape[1], frame.shape[0])
        cv.moveWindow('Select Bounding Box', 50, 50)
        bbox = cv.selectROI('Select Bounding Box', frame, False)
        return bbox

    # ***** Inicializacija sledilcev ***************************************
    trackers = []
    for i in range(trackerNo):
        tracker = trackerCreator(trackerArray[i])
        trackers.append(tracker)

    # ***** Oznacevanje tarc na prvi sliki sekvence ************************
    bboxes = []   # ODKOMENTIRAJ SPODNJE TRI VRSTICE ZA ROCNO IZBIRO TARCE
    #for i in range(trackerNo):
    #    bbox_cur = boundingBoxSelect()
    #    bboxes.append(bbox_cur)
    bboxes.append(bbox0)

    bbox_cur = bbox0

    #***** Zapis podatkov za prvi frame v test_result
    newLine = str(frameCounter) + ' ' + str(int(bbox_cur[0])) + ' ' + str(int(bbox_cur[1])) + ' ' + str(int(bbox_cur[2])) + ' ' + str(int(bbox_cur[3])) + ' ' + str(0) + '\n'
    test_result.write(newLine)

    # ***** Zacetek sledenja ***********************************************
    for i in range(trackerNo):
        tr_cur = trackers[i]
        #bb_cur = bboxes[i]
        #ok = tr_cur.init(frame, bb_cur)
        ok = tr_cur.init(frame, bbox0)

    #seq = cv.VideoCapture(path_source)

    # ***** Inicializacija seznama za shranjevanje trenutnih vrednosti FPS *
    fps_list = []

    while True:
    # ***** Zopet beremo slike tokrat od druge naprej **********************
        ok, frame = seq.read()
        if not ok:
            test_result.close()
            # izpis povprečnega FPS rate-a ko sledilec pride do konca sekvence:
            print(tr_input_type, "FPS_avg: ", round(np.average(fps_list), 2))
            break

        frameCounter += 1
        remainder = frame_Num - frameCounter

    #***** Branje vrstic Ground_truth-a za primerjavo Bounding box-ov ******
        line_cur = gt.readline().split(',')
        bbox_cur = (int(line_cur[1]), int(line_cur[2]), int(line_cur[3]), int(line_cur[4]))
        p1_cur = (int(bbox_cur[0]), int(bbox_cur[1]))
        p2_cur = (int(bbox_cur[0] + bbox_cur[2]), int(bbox_cur[1] + bbox_cur[3]))
        cv.rectangle(frame, p1_cur, p2_cur, (0, 0, 255), 2, 1)


    #***** Zacetek casovnika na podlagi katerega se racuna FPS *************
        timer = cv.getTickCount()

    #***** Posodobitev vseh sledilcev **************************************
        for i in range(trackerNo):
            tr_cur = trackers[i]
            ok, bb_cur = tr_cur.update(frame)
            bboxes[i] = bb_cur

    #***** Izracun FPS rate-a in dodajanje v seznam*************************
        fps = cv.getTickFrequency() / (cv.getTickCount() - timer);
        fps_list.append(fps)

    #***** Izris "bounding box-ov" in zapis njihovih tock v test_result ****
        if ok:  # tu moras preveriti se OK za ostale trackerje
            # Tracking successful:

            for i in range(trackerNo):
                bb_cur = bboxes[i]
                p1 = (int(bb_cur[0]), int(bb_cur[1]))
                p2 = (int(bb_cur[0] + bb_cur[2]), int(bb_cur[1] + bb_cur[3]))
                cv.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                newLine = str(frameCounter) + ' ' + str(int(bb_cur[0])) + ' ' + str(int(bb_cur[1])) + ' ' + str(int(bb_cur[2])) + ' ' + str(int(bb_cur[3])) + ' ' + str(0) + '\n'
                test_result.write(newLine)

        else:
            # Tracking failure
            cv.putText(frame, "Tracking failure detected", (100, 80), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            # naslednji ukaz zapisuje zadnje znane parametre bounding box-a uspesno trackane tarce z zadnjim flag-om na 1, kar pomeni izgubljeno sled
            newLine = str(frameCounter) + ' ' + str(int(bb_cur[0])) + ' ' + str(int(bb_cur[1])) + ' ' + str(int(bb_cur[2])) + ' ' + str(int(bb_cur[3])) + ' ' + str(1) + '\n'
            test_result.write(newLine)

    #***** Prikaz sledilca z vsemi "bounding box-i" ************************
        for i in range(trackerNo):
            tr_cur = trackerArray[i]
            cv.putText(frame, tr_cur + " Tracker", (100, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display FPS on frame:
        cv.putText(frame, "FPS : " + str(int(fps)), (100, 80), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    #***** Prikaz preostalih slicic v sekvenci *****************************
        cv.putText(frame, "Remaining frames: " + str(int(remainder)), (100, 50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    #***** Prikaz rezultatov v oknu ****************************************
        cv.namedWindow('Tracking', cv.WINDOW_NORMAL)
        cv.resizeWindow('Tracking', frame.shape[1], frame.shape[0])
        cv.moveWindow('Tracking', 50, 50)
        cv.imshow("Tracking", frame)

        if frameCounter < 3:
            time.sleep(0.5)

    #***** Izhod v primeru pritiska na ESC tipko ***************************
        k = cv.waitKey(1) & 0xff
        if k == 27:
            cv.destroyAllWindows()
            test_result.close()
            break
