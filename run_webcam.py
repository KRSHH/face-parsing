import cv2
import torch
import torchvision.transforms as transforms
from models.bisenet import BiSeNet
from utils.common import vis_parsing_maps

def prepare_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    image_tensor = transform(image)
    image_batch = image_tensor.unsqueeze(0)
    return image_batch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 19
    model = BiSeNet(num_classes, backbone_name="resnet18")
    model.to(device)
    model.load_state_dict(torch.load("./weights/resnet18.pt"))
    model.eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))
        image_batch = prepare_image(image).to(device)

        with torch.no_grad():
            output = model(image_batch)[0]
            predicted_mask = output.squeeze(0).cpu().numpy().argmax(0)

        vis_image = vis_parsing_maps(image, predicted_mask)
        cv2.imshow("Face Segmentation", vis_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
