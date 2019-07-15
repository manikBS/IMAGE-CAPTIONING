def define_parinject_model(vocab_size, max_length):
    # feature extractor model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    fe3 = fe2
    for i in range(33): 
        fe3=concatenate([fe3,fe2],axis=1)
    fe4 = Reshape((34,256))(fe3)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=False)(inputs2)
    se2 = Dropout(0.5)(se1)
    #se3=Flatten()(se2)
    #encoder
    encode1=concatenate([fe4,se2],axis=-1)
    #encode2 = Reshape((35,256))(encode1)
    se3 = LSTM(256)(encode1)
    # decoder model
    decoder1 = Dense(256, activation='relu')(se3)
    outputs = Dense(vocab_size, activation='softmax')(decoder1)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    return model
