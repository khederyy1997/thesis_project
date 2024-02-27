from xml.dom import minidom
import os

class PML_generator:
    def __init__(self):

        self.root = minidom.Document()

        xml = self.root.createElement('pml')

        header = self.root.createElement('header')
        source = self.root.createElement('source')
        sourceText = self.root.createTextNode("test")
        source.appendChild(sourceText)
        timeStamps = self.root.createElement('timeStamps')
        timeStampsText = self.root.createTextNode("0")
        timeStamps.appendChild(timeStampsText)
        header.appendChild(source)
        header.appendChild(timeStamps)
        xml.appendChild(header)

        person = self.root.createElement("person")
        person.setAttribute("id", "test")

        sensingLayer = self.root.createElement("sensingLayer")

        self.headPose = self.root.createElement("headPose")
        self.headRotation = self.root.createElement("rotation")
        self.headPose.appendChild(self.headRotation)
        self.SetHeadRotation(0, 0, 0)

        self.headPosition = self.root.createElement("position")
        self.headPose.appendChild(self.headPosition)
        self.SetHeadPosition(0, 0, 0)

        headPoseConfidence = self.root.createElement("confidence")
        self.headPoseConfidenceText = self.root.createTextNode("1.0")
        headPoseConfidence.appendChild(self.headPoseConfidenceText)

        self.headPose.appendChild(headPoseConfidence)
        sensingLayer.appendChild(self.headPose)

        self.handsPose = self.root.createElement("handsPose")
        self.leftHand = self.root.createElement("left")
        self.rightHand = self.root.createElement("right")
        self.handsPose.appendChild(self.leftHand)
        self.handsPose.appendChild(self.rightHand)
        sensingLayer.appendChild(self.handsPose)

        self.leftHandPosition = self.root.createElement("position")
        self.leftHandPosition.setAttribute("x", "0.0")
        self.leftHandPosition.setAttribute("y", "0.0")
        self.leftHandPosition.setAttribute("z", "0.0")
        self.rightHandPosition = self.root.createElement("position")
        self.rightHandPosition.setAttribute("x", "0.0")
        self.rightHandPosition.setAttribute("y", "0.0")
        self.rightHandPosition.setAttribute("z", "0.0")
        self.leftHandConfidence = self.root.createElement("confidence")
        self.leftHandConfidenceText = self.root.createTextNode("0.0")
        self.leftHandConfidence.appendChild(self.leftHandConfidenceText)
        self.rightHandConfidence = self.root.createElement("confidence")
        self.rightHandConfidenceText = self.root.createTextNode("0.0")
        self.rightHandConfidence.appendChild(self.rightHandConfidenceText)
        self.leftHand.appendChild(self.leftHandPosition)
        self.leftHand.appendChild(self.leftHandConfidence)
        self.rightHand.appendChild(self.rightHandPosition)
        self.rightHand.appendChild(self.rightHandConfidence)

        self.gaze = self.root.createElement("gaze")
        self.gazeH = self.root.createElement("horizontal")
        self.gazeHText = self.root.createTextNode("0")
        self.gazeH.appendChild(self.gazeHText)
        self.gaze.appendChild(self.gazeH)
        self.gazeV = self.root.createElement("vertical")
        self.gazeVText = self.root.createTextNode("0")
        self.gazeV.appendChild(self.gazeVText)
        self.gaze.appendChild(self.gazeV)
        target = self.root.createElement("target")
        targetText = self.root.createTextNode("away")
        target.appendChild(targetText)
        self.gaze.appendChild(target)
        sensingLayer.appendChild(self.gaze)


        person.appendChild(sensingLayer)

        behaviorLayer = self.root.createElement("behaviorLayer")

        person.appendChild(behaviorLayer)

        xml.appendChild(person)

        self.root.appendChild(xml)

        # xml_str = self.root.toprettyxml(indent="\t")

    def SetHeadRotation(self, headRotX, headRotY, headRotZ):
        self.headRotX = headRotX
        self.headRotY = headRotY
        self.headRotZ = headRotZ
        self.headRotation.setAttribute("rotX", str(self.headRotX))
        self.headRotation.setAttribute("rotY", str(self.headRotY))
        self.headRotation.setAttribute("rotZ", str(self.headRotZ))

    def SetHeadPosition(self, headPosX, headPosY, headPosZ):
        self.headPosX = headPosX
        self.headPosY = headPosY
        self.headPosZ = headPosZ
        self.headPosition.setAttribute("x", str(self.headPosX))
        self.headPosition.setAttribute("y", str(self.headPosY))
        self.headPosition.setAttribute("z", str(self.headPosZ))

    def SetHeadConfidence(self, confidence):
        self.headPoseConfidenceText.nodeValue = confidence

    def SetHandsPosition(self, lX, lY, lZ, lC, rX, rY, rZ, rC):
        self.leftHandPosition.setAttribute("x", str(lX))
        self.leftHandPosition.setAttribute("y", str(lY))
        self.leftHandPosition.setAttribute("z", str(lZ))
        self.leftHandConfidenceText.nodeValue = str(lC)
        self.rightHandPosition.setAttribute("x", str(rX))
        self.rightHandPosition.setAttribute("y", str(rY))
        self.rightHandPosition.setAttribute("z", str(rZ))
        self.rightHandConfidenceText.nodeValue = str(rC)

    def SetGaze(self, gazeH, gazeV):
        self.gazeH = gazeH
        self.gazeV = gazeV
        self.gazeHText.nodeValue = self.gazeH
        self.gazeVText.nodeValue = self.gazeV

    def toString(self, pretty=False):
        if not pretty:
            return self.root.toxml()
        else:
            return self.root.toprettyxml()