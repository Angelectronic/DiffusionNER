import torch

from diffusionner import util
import random

def create_train_sample(doc, repeat_gt_entities = 100):
    encodings = doc.encoding
    seg_encoding = doc.seg_encoding

    token_count = len(doc.tokens)
    context_size = len(encodings)

    context2token_masks = []
    for t in doc.tokens:
        context2token_masks.append(create_entity_mask(*t.span, context_size))

    gt_entities_spans_token = []
    gt_entity_types = []
    gt_entity_masks = []
    
    for e in doc.entities:
        gt_entities_spans_token.append(e.span_token)
        gt_entity_types.append(e.entity_type.index)
        gt_entity_masks.append(1)

    for e in doc.pred_entities:
        gt_entities_spans_token.append(e.span_token)
        gt_entity_types.append(e.entity_type.index)
        gt_entity_masks.append(0)

    total_gt_entities = len(gt_entities_spans_token)

    if repeat_gt_entities != -1:
        if total_gt_entities!=0:
            k = repeat_gt_entities//total_gt_entities
            m = repeat_gt_entities%total_gt_entities
            gt_entities_spans_token = gt_entities_spans_token*k + gt_entities_spans_token[:m]
            gt_entity_types = gt_entity_types*k + gt_entity_types[:m]
            gt_entity_masks = gt_entity_masks*k + gt_entity_masks[:m]
            assert len(gt_entities_spans_token) == len(gt_entity_types) == len(gt_entity_masks) == repeat_gt_entities

    # create tensors
    # token indices
    encodings = torch.tensor(encodings, dtype=torch.long)
    seg_encoding = torch.tensor(seg_encoding, dtype=torch.long)

    # masking of tokens
    context_masks = torch.ones(context_size, dtype=torch.bool)

    # context_masks = torch.tensor(seg_encoding, dtype=torch.bool)
    token_masks = torch.ones(token_count, dtype=torch.bool)

    # also create samples_masks:
    context2token_masks = torch.stack(context2token_masks)

    if len(gt_entity_types) > 0:
        gt_entity_types = torch.tensor(gt_entity_types, dtype=torch.long)
        # gt_entity_spans_token = torch.tensor(gt_entities_spans_token, dtype=torch.float) / len(doc.tokens)
        gt_entity_spans_token = torch.tensor(gt_entities_spans_token, dtype=torch.long)
        gt_entity_masks = torch.tensor(gt_entity_masks, dtype=torch.bool)
    else:
        gt_entity_types = torch.zeros([1], dtype=torch.long)
        gt_entity_spans_token = torch.zeros([1, 2], dtype=torch.long)
        gt_entity_masks = torch.zeros([1], dtype=torch.bool)
    return dict(encodings=encodings, context_masks=context_masks, seg_encoding = seg_encoding, context2token_masks=context2token_masks, token_masks=token_masks, gt_types=gt_entity_types, gt_spans=gt_entity_spans_token, entity_masks=gt_entity_masks, meta_doc = doc)


def create_eval_sample(doc, processor = None, repeat_gt_entities = 100):
    # if len(doc.encoding) > 512:
    #     return None
    encodings = doc.encoding
    seg_encoding = doc.seg_encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)
    
    # import pdb; pdb.set_trace()
    # all tokens
    context2token_masks = []
    for t in doc.tokens:
        context2token_masks.append(create_entity_mask(*t.span, context_size))

    gt_entities_spans_token = []
    gt_entity_types = []
    gt_entity_masks = []

    for e in doc.pred_entities:
        gt_entities_spans_token.append(e.span_token)
        gt_entity_types.append(e.entity_type.index)
        gt_entity_masks.append(0)

    total_gt_entities = len(gt_entities_spans_token)

    if repeat_gt_entities != -1:
        if total_gt_entities!=0:
            k = repeat_gt_entities//total_gt_entities
            m = repeat_gt_entities%total_gt_entities
            gt_entities_spans_token = gt_entities_spans_token*k + gt_entities_spans_token[:m]
            gt_entity_types = gt_entity_types*k + gt_entity_types[:m]
            gt_entity_masks = gt_entity_masks*k + gt_entity_masks[:m]
            assert len(gt_entities_spans_token) == len(gt_entity_types) == len(gt_entity_masks) == repeat_gt_entities

    # create tensors
    # token indices
    encodings = torch.tensor(encodings, dtype=torch.long)
    seg_encoding = torch.tensor(seg_encoding, dtype=torch.long)

    
    # masking of tokens
    context_masks = torch.ones(context_size, dtype=torch.bool)

    token_masks = torch.ones(token_count, dtype=torch.bool)

    # also create samples_masks:
    # tensors to mask entity/relation samples of batch
    # since samples are stacked into batches, "padding" entities/relations possibly must be created
    # these are later masked during loss computation
    context2token_masks = torch.stack(context2token_masks)
    if len(gt_entity_types) > 0:
        gt_entity_types = torch.tensor(gt_entity_types, dtype=torch.long)
        # gt_entity_spans_token = torch.tensor(gt_entities_spans_token, dtype=torch.float) / len(doc.tokens)
        gt_entity_spans_token = torch.tensor(gt_entities_spans_token, dtype=torch.long)
        gt_entity_masks = torch.tensor(gt_entity_masks, dtype=torch.bool)
    else:
        gt_entity_types = torch.zeros([1], dtype=torch.long)
        gt_entity_spans_token = torch.zeros([1, 2], dtype=torch.long)
        gt_entity_masks = torch.zeros([1], dtype=torch.bool)
    return dict(encodings=encodings, context_masks=context_masks, seg_encoding = seg_encoding, context2token_masks=context2token_masks, token_masks=token_masks, meta_doc = doc, gt_types=gt_entity_types, gt_spans=gt_entity_spans_token, entity_masks=gt_entity_masks)

def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end+1] = 1
    return mask

def collate_fn_padding(batch, args):
    batch = list(filter(lambda x: x is not None, batch))
    padded_batch = dict()
    keys = batch[0].keys()
    number_cut_off_span = random.randint(0, len(batch[0]["gt_spans"])-1) if len(batch[0]["gt_spans"]) > 0 else 0
    for key in keys:
        samples = [s[key] for s in batch]
        if key.startswith("meta"):
            padded_batch[key] = samples
            continue

        if key.startswith("image_inputs"):
            if batch[0]["image_inputs"] == None:
                padded_batch["image_inputs"] = None
            else:
                padded_batch["image_inputs"] = dict((k , torch.cat([s["image_inputs"][k] for s in batch], dim=0) ) for k in batch[0]["image_inputs"].keys())
            continue
        
        if batch[0][key] is None:
            padded_batch[key] = None
            continue
        
        if (key == "gt_spans" or key == "gt_types" or key == "entity_masks") and args.dynamic_k:
            max_len = number_cut_off_span
            tensors = []
            for s in batch:
                data = s[key]
                if data.shape[0] >= max_len:
                    data = data[:max_len]
                else:
                    pad_len = max_len - data.shape[0]
                    pad = torch.zeros((pad_len,) + data.shape[1:], dtype=data.dtype, device=data.device)
                    data = torch.cat([data, pad], dim=0)
                tensors.append(data)
            padded_batch[key] = torch.stack(tensors)
            padded_batch["dynamic_k"] = torch.tensor(number_cut_off_span, dtype=torch.int)
            continue

        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = util.padded_stack([s[key] for s in batch])

    return padded_batch
