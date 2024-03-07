from coretex import CoretexSegmentationInstance, CoretexImageAnnotation, BBox, ImageDatasetClass


def getClassAnnotation(annotation: CoretexImageAnnotation, class_: ImageDatasetClass) -> CoretexSegmentationInstance:
    for instance in annotation.instances:
        if instance.classId in class_.classIds:
            return instance

    raise ValueError(f"Failed to find \"{class_.label}\" annotation")


def offsetSegmentations(instance: CoretexSegmentationInstance, offsetX: int, offsetY: int) -> None:
    for segmentation in instance.segmentations:
        for i, p in enumerate(segmentation):
            if i % 2 == 0:
                segmentation[i] = p + offsetX
            else:
                segmentation[i] = p + offsetY

    instance.bbox = BBox.fromPoly([e for sublist in instance.segmentations for e in sublist])


def resizePolygon(polygon: list[int], oldWidth: int, oldHeight: int, newWidth: int, newHeight: int) -> list[int]:
    resized: list[int] = []

    for i, value in enumerate(polygon):
        if i % 2 == 0:
            resized.append(int(value / oldWidth * newWidth))
        else:
            resized.append(int(value / oldHeight * newHeight))

    return resized


def resizeInstance(
    instance: CoretexSegmentationInstance,
    oldWidth: int,
    oldHeight: int,
    newWidth: int,
    newHeight: int
) -> CoretexSegmentationInstance:

    resizedSegmentations: list[list[int]] = []

    for segmentation in instance.segmentations:
        resizedSegmentations.append(resizePolygon(segmentation, oldWidth, oldHeight, newWidth, newHeight))

    return CoretexSegmentationInstance.create(
        instance.classId,
        BBox.fromPoly([e for sublist in resizedSegmentations for e in sublist]),
        resizedSegmentations
    )
