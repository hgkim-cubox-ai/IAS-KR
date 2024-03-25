
# parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer'])
# parser.add_argument('--seed', type=int, default=777)
# parser.add_argument('--save_path', type=str, default='results/tmp')
# parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu', 'ddp'])
# # Data
# parser.add_argument('--data_path', type=str, default='C:/Users/heegyoon/Desktop/data')
# parser.add_argument('--train_sets', type=str, default=['mnist',])
# parser.add_argument('--val_sets', type=str, default=None)
# parser.add_argument('--test_sets', type=str, default=['mnist',])
# parser.add_argument('--batch_size', type=int, default=256)
# parser.add_argument('--num_workers', type=int, default=8)
# # Architecture
# parser.add_argument('--model', type=str, default='ae',
#                     choices=['ae', 'vae'])
# parser.add_argument('--latent_dim', type=int, default=2)
# parser.add_argument('--hidden_dims', type=int, default=[512,512,32], nargs='+')
# # Train
# # parser.add_argument('--loss_fns', type=str, default=None)
# # parser.add_argument('--num_epochs', type=int, default=30)
# # parser.add_argument('--loss_fn', type=str, default='mse',
# #                     choices=['mse','crossentropy'])
# parser.add_argument('--loss_fns', type=str, nargs='+')
# parser.add_argument('--optimizer', type=str, default='adam',
#                     choices=['adam','sgd'])
# parser.add_argument('--base_lr', type=float, default=1e-3)
# parser.add_argument('--lr_decay_epoch', type=int, default=10)
# # etc
# parser.add_argument('--trained_model_path', type=str)

# args = parser.parse_args()
# print('')
# Debugging
# args.mode = 'infer'
# args.num_workers = 1
# args.latent_dim = 2
# args.trained_model_path = 'results/AE_z2/saved_models/best_100.pth'
# args.batch_size = 1