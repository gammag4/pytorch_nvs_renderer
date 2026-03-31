
otimizacoes

a ideia era fazer algo que usava cuda streams pra despachar em paralelo multiplos frames pra aumentar o fps,
mas a gpu ja esta saturada do jeito que ta (100% uso, mesmo que nao use 100% da memoria)

tentei usar cuda streams com queue que vai schedulando gerar novos frames,
mas como era de se esperar pelo fato de a gpu estar saturada,
o fps caiu por causa do tamanho


tested tensorrt didnt make a difference
maybe i did it wrong
