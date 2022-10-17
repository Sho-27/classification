import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

acc = [0.6296296119689941, 0.6203703880310059, 0.7222222089767456, 0.6851851940155029, 0.7777777910232544, 0.7314814925193787, 0.7777777910232544, 0.7129629850387573, 0.7314814925193787, 0.7685185074806213, 0.8240740895271301, 0.8981481194496155, 0.9351851940155029, 0.9629629850387573, 0.8518518805503845, 0.8518518805503845, 0.7685185074806213, 0.7407407164573669, 0.7592592835426331, 0.8796296119689941]
epoch = range(1, len(acc) + 1)
loss = [2691.034423828125, 2690.57861328125, 2982.9970703125, 2209.380615234375, 1388.530517578125, 679.8135375976562, 523.0442504882812, 489.07476806640625, 485.2402648925781, 473.5234069824219, 204.1183624267578, 184.3557586669922, 89.14727020263672, 56.39421081542969, 40.792694091796875, 40.97770690917969, 16.697650909423828, 43.66401672363281, 42.49100875854492, 23.486459732055664]

Figure = plt.figure(figsize = (8, 6))

Figure.subplots_adjust(hspace = 0.6, wspace = 0.4)
acc_graph = Figure.add_subplot(2, 1, 1, title = 'Accuracy_graph', xlabel = 'Epoch', ylabel = 'Accuracy')
loss_graph = Figure.add_subplot(2, 1, 2, title = 'Loss_graph', xlabel = 'Epoch', ylabel = 'Loss')

acc_graph.plot(epoch, acc)
loss_graph.plot(epoch, loss)

acc_graph.xaxis.set_major_locator(ticker.MultipleLocator(1))
loss_graph.xaxis.set_major_locator(ticker.MultipleLocator(1))
loss_graph.yaxis.set_major_formatter(ticker.FuncFormatter(lambda epoch, loss : '{:,}'.format(int(epoch))))

plt.show()