attr_dict = {}


attr_dict['ffhq_20_USER'] = ["man", "woman", "child", "arched eyebrows", "smiling", "wearing hat", "wearing eyeglasses", "wearing necklace", "blonde hair","black hair", \
                             "bangs","gray hair", "wavy hair", "bald head","red lip", "makeup", "wearing earlings","beard","double chin", "young"]

                             
#gpt
# attr_dict['ffhq_alloc2'] = ['clean-shaven', 'beard', 'mustache', 'Wide-eyed', 'thin lips','bald', 'Glasses-wearing', 'Freckled', 'almond-shaped eyes', 'Scarred', 
# 'Wrinkled', 'soul patch', 'high forehead','hooded eyes', 'piercings', 'prominent cheekbones', 'full lips', 'braided','Upturned-nosed', 'Youthful']


attr_dict['celebA_20_USER'] = ["man", "woman", "child", "arched eyebrows", "smiling", "wearing hat", "wearing eyeglasses", "wearing necklace", "blonde hair","black hair", \
                             "bangs","gray hair", "wavy hair", "bald head","red lip", "makeup", "wearing earlings","beard","double chin", "young"]

attr_dict['lsuncat_20_USER'] = ['green eyes', 'blue eyes', 'pink nose', 'black fur', 'white fur', 'yellow eyes', 'fluffy fur', 'solid color', 'black nose',  'tabby pattern', 
'white paws', 'gray fur', 'brown fur',   'round eyes', 'calico pattern', 'orange fur', 'pointed ears', 'tufted ears', 'slanted eyes', 'long tail']

attr_dict['lsuncat_20_color'] = ['green fur', 'blue fur', 'white fur', 'black fur', 'yellow fur', 'dotted fur', 'ginger fur', 'orange fur', 'tabby fur', 'siamese fur', 'tortoiseshell fur',
'calcico fur', 'cream fur', 'fawn fur', 'lilac fur', 'chocolate fur', 'red fur', 'sable fur', 'striped fur', 'navy fur']

attr_dict['lsuncat_20_shape'] = ['fluffy fur',  'long eyelashes', 'round eyes',  'pointed ears', 'tufted ears', 'slanted eyes', 'long tail',
 'long fur',  'short fur', 'small ears', 'small nose','Almond-shaped eyes','round face', 'floppy ears','long whiskers', 'white chin', 'amber eyes',
'hazel eyes', 'Wide-set eyes',  'short tail']



attr_dict['coco_20_USER']=['man', 'woman', 'he', 'people', 'person', 'table', 'group', 'street', 'water', 'plate', 
                           'cat', 'field', 'couple', 'dog', 'food', 'beach', 'bed', 'bathroom', 'pizza', 'grass']

attr_dict['coco_30_USER']=['man', 'woman', 'he', 'people', 'person', 'table', 'group', 'street', 'water', 'plate', 
                           'cat', 'field', 'couple', 'dog', 'food', 'beach', 'bed', 'bathroom', 'pizza', 'grass',
                           'kitchen', 'skateboard', 'picture', 'road', 'train', 'building', 'snow', 'surfboard', 'toilet', 'giraffe']

attr_dict['coco_40_USER']=['man', 'woman', 'he', 'people', 'person', 'table', 'group', 'street', 'water', 'plate', 
                           'cat', 'field', 'couple', 'dog', 'food', 'beach', 'bed', 'bathroom', 'pizza', 'grass',
                           'kitchen', 'skateboard', 'picture', 'road', 'train', 'building', 'snow', 'surfboard', 'toilet', 'giraffe',
                           'room', 'men', 'bunch', 'ball', 'air', 'bench', 'clock', 'boy', 'sign', 'tree']





# #gpt 40
# attr_dict['ffhq_alloc30']=['clean-shaven', 'beard', 'mustache', 'Wide-eyed', 'thin lips', 'bald', 'Glasses-wearing', 'Freckled', 'almond-shaped eyes','Scarred', 'Wrinkled', 'soul patch', 'high forehead','hooded eyes', 'piercings',
#        'prominent cheekbones', 'full lips', 'braided', 'Upturned-nosed', 'Youthful', 'approachable', 'arched eyebrows', 'Thin-lipped','Thin-eyebrowed', 'birthmark', 'bobbed', 'composed',
#        'curly hair', 'deep-set eyes', 'Thick-eyebrowed']


# attr_dict['lsun_cat_alloc30']=['green eyes', 'blue eyes', 'pink nose', 'black fur', 'white fur', 'yellow eyes', 'fluffy fur', 'solid color', 'black nose',  'tabby pattern', 
# 'white paws', 'gray fur', 'brown fur',   'round eyes', 'calico pattern', 'orange fur', 'pointed ears', 'tufted ears', 'slanted eyes', 'long tail',
# 'spotted fur','ginger fur', 'long fur',  'short fur','tortoiseshell pattern', 'small ears', 'small nose','Almond-shaped eyes', 'black paws', 'round face']


# attr_dict['metface_alloc30']=['Full lips', 'Double chin', 'Deep-set eyes', 'Wide-set eyes', 'Close-set eyes', 'Pointed chin', 'Thick eyelashes','Hooded eyelids', 'Sideburns', 'Rosy cheeks', 'Bulbous nose',
# 'High cheekbones', 'Goatee', 'Thick eyebrows', 'Laugh lines', 'Moustache', 'Almond-shaped eyes', 'Full cheeks', 'Beard', 'Arched eyebrows',
# 'Thin eyebrows', 'Sparse eyelashes', 'Curved eyebrows', 'Crooked nose', 'Facial hair', 'Small nose', 'Wrinkles on forehead', 'Square jaw', 'Long eyelashes', 'Dimples']




#gpt60
# attr_dict['ffhq_alloc60']=['clean-shaven', 'beard', 'mustache', 'Wide-eyed', 'thin lips', 'bald', 'Glasses-wearing', 'Freckled', 'almond-shaped eyes','Scarred', 'Wrinkled', 'soul patch', 'high forehead','hooded eyes', 'piercings',
#        'prominent cheekbones', 'full lips', 'braided', 'Upturned-nosed', 'Youthful', 'approachable', 'arched eyebrows', 'Thin-lipped','Thin-eyebrowed', 'birthmark', 'bobbed', 'composed',
#        'curly hair', 'deep-set eyes', 'Thick-eyebrowed', 'earrings', 'eyebrow thickness', 'facial hair','goatee', 'heart-shaped face', 'long eyelashes', 'low forehead','monolid eyes', 'nasolabial folds',
#        'diamond-shaped face','Small-nosed', 'Tattooed', 'Hairy', 'Flat-nosed', 'Female','Elderly', 'Double-chinned', 'Chiseled', 'Bulbous-nosed', 'Broken-nosed', 'Big-eared''Angular', 'Angry', 'Full-lipped', 
#        'Happy', 'Square-jawed', 'High-cheekboned','Pointed-eared',  'Oval-shaped','Narrow-eyed', "Male"]

# attr_dict['lsun_cat_alloc60']=['green eyes', 'blue eyes', 'pink nose', 'black fur', 'white fur', 'yellow eyes', 'fluffy fur', 'solid color', 'black nose',  'tabby pattern', 
# 'white paws', 'gray fur', 'brown fur',   'round eyes', 'calico pattern', 'orange fur', 'pointed ears', 'tufted ears', 'slanted eyes', 'long tail',
# 'spotted fur','ginger fur', 'long fur',  'short fur','tortoiseshell pattern', 'small ears', 'small nose','Almond-shaped eyes', 'black paws', 'round face', 'white chest',
# 'floppy ears', 'black and white pattern', 'striped tail','striped pattern', 'siamese pattern', 'white belly','long whiskers', 'white chin', 'amber eyes',
# 'hazel eyes', 'Wide-set eyes',  'black spots', 'short tail', 'whiskers', 'gold eyes', 'big ears','unkempt', 'close-set eyes', 'Heterochromatic eyes','big nose', 'fluffy tail',  
# 'folded ears', 'alert', 'black', 'White tip on tail', 'Wide pupils', 'black and brown coat', 'angular cheeks', 'black paw pads']


# attr_dict['metface_alloc60']= ['Full lips', 'Double chin', 'Deep-set eyes', 'Wide-set eyes', 'Close-set eyes', 'Pointed chin', 'Thick eyelashes','Hooded eyelids', 'Sideburns', 'Rosy cheeks', 'Bulbous nose',
# 'High cheekbones', 'Goatee', 'Thick eyebrows', 'Laugh lines', 'Moustache', 'Almond-shaped eyes', 'Full cheeks', 'Beard', 'Arched eyebrows',
# 'Thin eyebrows', 'Sparse eyelashes', 'Curved eyebrows', 'Crooked nose', 'Facial hair', 'Small nose', 'Wrinkles on forehead', 'Square jaw', 'Long eyelashes', 'Dimples',
# 'Straight nose', 'Thin lips', 'Straight eyebrows', 'Dark-colored eyes', 'Small ears', 'Upturned nose','Sharp jawline', 'Thin nostrils', 'Small pupils',
# 'Wrinkles around eyes', 'High forehead', 'Piercing eyes','Light-colored eyes', 'Pronounced nasolabial folds','Pronounced philtrum', 'Hollow cheeks', 'Large pupils',
# 'Prominent eyebrows', 'Large forehead', 'Round jaw','Round nostrils', 'Clean-shaven', 'Hooked nose', 'Uneven irises','Tattoo on neck', 'Long nose', 'Prominent brow ridge', 'Low ears',
# 'Light-colored irises', 'Mole above lip']
