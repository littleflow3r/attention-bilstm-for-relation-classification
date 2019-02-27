import torch
import torch.nn as nn
import torch.nn.functional as F

class attbilstm(nn.Module):
    def __init__(self, vocab_size, config, vec=None):
        super().__init__()
        self.hidden_dim = config['hidden_dim']
        self.batch_size = config['batch_size']
        self.emb_dim = config['emb_dim']
        self.gpu = config['gpu']
        
        self.embedding = nn.Embedding(vocab_size, config['emb_dim'])
        if vec is not None:
            self.embedding.weight.data.copy_(vec) #load pretrained
            #self.embedding.weight.requires_grad = False #non-trainable
        self.encoder = nn.LSTM(config['emb_dim'], config['hidden_dim'], num_layers=config['nlayers'], bidirectional=config['bidir'], dropout=config['dropout'])
        self.fc = nn.Linear(config['hidden_dim'], config['out_dim'])
        self.dropout = nn.Dropout(config['dropout'])
        #self.hidden = nn.Parameters(self.batch_size, self.hidden_dim)
    
    def attnetwork(self, encoder_out, final_hidden):
        hidden = final_hidden.squeeze(0)
        #M = torch.tanh(encoder_out)
        attn_weights = torch.bmm(encoder_out, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden = torch.bmm(encoder_out.transpose(1,2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        #print (wt.shape, new_hidden.shape)
        #new_hidden = torch.tanh(new_hidden)
        #print ('UP:', new_hidden, new_hidden.shape)
        
        return new_hidden
     
    def attnetwork1(self, encoder_out, final_hidden):
        print ('ddddd')
        #print (encoder_out.size())
        print ('fh:', final_hidden)
        merged_state = torch.cat([s for s in final_hidden],1)
        merged_state = (merged_state.unsqueeze(2))
        #print (merged_state.size())
        encoder_out = encoder_out.permute(1,0,2)
        print (encoder_out.size(), merged_state.size())
        weights = torch.bmm(encoder_out, merged_state)
        weights = F.softmax(weights.squeeze(2)).unsqueeze(2)
        new_hidden = torch.bmm(torch.transpose(encoder_out,1,2), weight).squeeze(2)
        print ('BOTTOM:',new_hidden, new_hidden.shape)
        return new_hidden
    
    def forward(self, sequence):
        emb_input = self.embedding(sequence)    
        inputx = self.dropout(emb_input)
        output, (hn, cn) = self.encoder(inputx)
        fbout = output[:, :, :self.hidden_dim]+ output[:, :, self.hidden_dim:] #sum bidir outputs F+B
        fbout = fbout.permute(1,0,2)
        fbhn = (hn[-2,:,:]+hn[-1,:,:]).unsqueeze(0)
        #print (fbhn.shape, fbout.shape)
        attn_out = self.attnetwork(fbout, fbhn)
        #attn1_out = self.attnetwork1(output, hn)
        logits = self.fc(attn_out)
        return logits
        
