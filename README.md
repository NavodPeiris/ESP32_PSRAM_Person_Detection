You can allocate memory on PSRAM for your Tensor Arena

```
if (tensor_arena == NULL) {
    //allocate memory for TensorArena on PSRAM
    tensor_arena = (uint8_t *) ps_malloc(kTensorArenaSize);
}
```