from detectron2.layers import batched_nms
import torch
# @gxl
def ml_nms_gxl(boxlist, nms_thresh_nom, nms_thresh_spc, special_classes,
               max_proposals=-1, score_field="scores", label_field="labels"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.
    
    Args:
        boxlist (detectron2.structures.Boxes): 
        nms_thresh (float): 
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str): 
    """
    if nms_thresh_nom <= 0 or nms_thresh_spc <= 0:
        return boxlist
    boxes = boxlist.pred_boxes.tensor
    scores = boxlist.scores
    labels = boxlist.pred_classes
    fake_labels = torch.zeros_like(labels)
    inds_spc_labels = []
    inds_nom_labels = []
    for ind, label in enumerate(labels):
        if label in special_classes:
            inds_spc_labels.append(ind)
        else:
            inds_nom_labels.append(ind)

    keep_nom = batched_nms(boxes[inds_nom_labels], scores[inds_nom_labels], fake_labels[inds_nom_labels], nms_thresh_nom)
    keep_spc = batched_nms(boxes[inds_spc_labels], scores[inds_spc_labels], fake_labels[inds_spc_labels], nms_thresh_spc)
    inds_keep_nom = []
    inds_keep_spc = []
    for ind in keep_nom:
        inds_keep_nom.append(inds_nom_labels[ind])
    for ind in keep_spc:
        inds_keep_spc.append(inds_spc_labels[ind])
    keep = inds_keep_nom + inds_keep_spc
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist


def mc_nms(boxlist, nms_thresh, max_proposals=-1,
           score_field="scores", label_field="labels"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.
    
    Args:
        boxlist (detectron2.structures.Boxes):
        nms_thresh (float):
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str):
    """
    if nms_thresh <= 0:
        return boxlist
    boxes = boxlist.pred_boxes.tensor
    scores = boxlist.scores
    labels = boxlist.pred_classes
    # @gxl
    # Use fake labe to instead of original label to do NMS for all classes
    # Changing the multi levels NMS to multi classes NMS
    fake_labels = torch.zeros_like(labels)

    keep = batched_nms(boxes, scores, fake_labels, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist
