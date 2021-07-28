import torch
import torch.nn as nn

from main.detr.models.transformer import TransformerDecoder, TransformerDecoderLayer, \
    TransformerEncoder, TransformerEncoderLayer


def SetUseMixedPrecision(Flag):
    global UseMixedPrecision
    UseMixedPrecision = Flag


# @torch.cuda.amp.autocast(enabled=UseMixedPrecision)
def Prepare2DPosEncoding(PosEncodingX, PosEncodingY, RowNo, ColNo):
    PosEncodingX = PosEncodingX[0:ColNo].unsqueeze(0)  # x=[1,..,20]
    PosEncodingY = PosEncodingY[0:RowNo]

    for i in range(RowNo):

        CurrentY = PosEncodingY[i, :].unsqueeze(0).unsqueeze(0).repeat(1, ColNo, 1)

        if i == 0:
            PosEncoding2D = torch.cat((PosEncodingX, CurrentY), 2)
        else:
            CurrentPosEncoding2D = torch.cat((PosEncodingX, CurrentY), 2)

            PosEncoding2D = torch.cat((PosEncoding2D, CurrentPosEncoding2D), 0)

    return PosEncoding2D


# @torch.cuda.amp.autocast(enabled=UseMixedPrecision)
def Prepare2DPosEncodingSequence(PosEncodingX, PosEncodingY, RowNo, ColNo):
    PosEncoding2D = Prepare2DPosEncoding(PosEncodingX, PosEncodingY, RowNo, ColNo)

    PosEncoding = PosEncoding2D.permute(2, 0, 1)
    PosEncoding = PosEncoding[:, 0:RowNo, 0:ColNo]
    PosEncoding = PosEncoding.reshape((PosEncoding.shape[0], PosEncoding.shape[1] * PosEncoding.shape[2]))
    PosEncoding = PosEncoding.permute(1, 0).unsqueeze(1)

    return PosEncoding


class DETR_Decoder(nn.Module):
    def __init__(self, NumClasses=68, K=512, EmbeddingMaxDim=16, LayersNo=2, HeadsNo=2):
        super(DETR_Decoder, self).__init__()

        # Positional Encoding
        self.PosEncodingX = nn.Parameter(torch.randn(EmbeddingMaxDim, int(K / 2)))
        self.PosEncodingY = nn.Parameter(torch.randn(EmbeddingMaxDim, int(K / 2)))

        # Encoder
        self.EncoderLayer = TransformerEncoderLayer(d_model=K, nhead=HeadsNo, dim_feedforward=int(K),
                                                    dropout=0.1, activation="relu", normalize_before=False)
        self.Encoder = TransformerEncoder(encoder_layer=self.EncoderLayer, num_layers=LayersNo)

        # Decoder
        self.DecoderLayer = TransformerDecoderLayer(d_model=K, nhead=HeadsNo, dim_feedforward=K,
                                                    dropout=0.1, activation="relu", normalize_before=False)
        self.Decoder = TransformerDecoder(decoder_layer=self.DecoderLayer, num_layers=LayersNo)

        # Decoder parameters ( Quries)
        self.DecoderQueriesPos = nn.Parameter(torch.randn(NumClasses, K))
        self.DecoderQueries = nn.Parameter(torch.randn(NumClasses, K))

    # @torch.cuda.amp.autocast(enabled=UseMixedPrecision)
    def forward(self, x, DropoutP=0.0):
        PosEncoding = Prepare2DPosEncodingSequence(self.PosEncodingX, self.PosEncodingY,
                                                   x.shape[2], x.shape[3])

        x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        x = x.permute(2, 0, 1)

        x = self.Encoder(src=x, pos=PosEncoding)

        DecoderQueries = self.DecoderQueries.unsqueeze(1).repeat(1, x.shape[1], 1)
        out = self.Decoder(tgt=DecoderQueries,
                           memory=x,
                           pos=PosEncoding,
                           query_pos=self.DecoderQueriesPos.unsqueeze(1))[0]

        out = out.permute(1, 2, 0)

        return out

        for p in self.parameters():
            print(p)
            print(p)
