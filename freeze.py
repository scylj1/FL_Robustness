from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
)

def freeze_unfreeze_layers(model, layer_indexs, unfreeze=False):
    if type(layer_indexs) == int:
      for name, param in model.named_parameters():
        #print(name)
        if param.requires_grad == True:
          if name.startswith(f'bert.encoder.layer.{layer_indexs}'):
            param.requires_grad_(unfreeze)
      print(f"successfully freeze layers index: {layer_indexs}")
        
    else:
      start = layer_indexs[0]
      end = layer_indexs[1]
      for name, param in model.named_parameters():
        if param.requires_grad == True:
          for i in range(start, end+1):
              if name.startswith(f'bert.encoder.layer.{i}'):
                param.requires_grad_(unfreeze)
      print(f"successfully freeze layers indexs from: {layer_indexs[0]} to: {layer_indexs[1]}, including {layer_indexs[1]}")

def get_para_num(model):
    lst = []
    for para in model.parameters():
        lst.append(para.nelement())
    print(f"total paras number: {sum(lst)}")

def get_trainable_para_num(model):
    lst = []
    for para in model.parameters():
        if para.requires_grad == True:
            lst.append(para.nelement())
    print(f"trainable paras number: {sum(lst)}")
    
def test():
    config = AutoConfig.from_pretrained('bert-base-uncased')
    model = AutoModelForMaskedLM.from_pretrained(
                'bert-base-uncased',
                from_tf=bool(".ckpt" in 'bert-base-uncased'),
                config=config,
            )
    get_para_num(model)
    get_trainable_para_num(model)

    #freeze_unfreeze_layers(model, 3, unfreeze=False)
    freeze_unfreeze_layers(model, (0,3), unfreeze=False)

    get_para_num(model)
    get_trainable_para_num(model)
    
if __name__ == "__main__":
    test()