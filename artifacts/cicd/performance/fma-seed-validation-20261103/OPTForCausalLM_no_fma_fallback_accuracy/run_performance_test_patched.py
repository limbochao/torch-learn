    def run_performance_test(
        self, name, model, example_inputs, optimize_ctx, experiment, tag=None
    ):
        def warmup(fn, model, example_inputs, mode, niters=5):
            peak_mem = 0
            start_stats = get_dynamo_stats()
            try:
                if current_device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()
                elif current_device == "npu":
                    torch_npu.npu.reset_peak_memory_stats()
                    torch_npu.npu.empty_cache()
                t0 = time.perf_counter()
                for _ in range(niters):
                    fn(model, example_inputs)
                t1 = time.perf_counter()
                latency = t1 - t0
                if current_device == "cuda":
                    peak_mem = get_peak_memory()
                elif current_device == "npu":
                    peak_mem = get_peak_memory_npu()
                elif current_device == "cpu":
                    total = psutil.virtual_memory().total
                    percentage = psutil.Process(os.getpid()).memory_percent()
                    peak_mem = percentage * total / 10**9
            except Exception as e:
                log.exception("Backend %s failed in warmup()", mode)
                raise RuntimeError(f"Backend {mode} failed in warmup()") from e
            dynamo_stats = get_dynamo_stats()
            dynamo_stats.subtract(start_stats)
            return latency, peak_mem, dynamo_stats

        # Cast the model to float16/float32 as necessary
        model, example_inputs = self.maybe_cast(model, example_inputs)

        model = self.deepcopy_and_maybe_ddp(model)

        self.init_optimizer(name, current_device, model.parameters())
        with self.pick_grad(name, self.args.training):
            ok, total = Stats.reset_counters()
            experiment_kwargs = {}
            if tag is not None:
                experiment_kwargs["tag"] = tag
            results = []
            eager_latency, eager_peak_mem, _ = warmup(
                self.model_iter_fn, model, example_inputs, "eager"
            )
            if getattr(self.args, "execution_mode", "both") == "eager":
                if not hasattr(model, name):
                    model.name = name
                return eager_performance_experiment(
                    self.args,
                    self.model_iter_fn,
                    model,
                    example_inputs,
                    tag=tag,
                    eager_peak_mem=eager_peak_mem,
                )

            optimized_model_iter_fn = optimize_ctx(self.model_iter_fn)
            dynamo_latency, dynamo_peak_mem, dynamo_stats = warmup(
                optimized_model_iter_fn, model, example_inputs, "dynamo"
            )

            if self.args.enable_profiler is not None:
                for phase, profile_fn in (("eager", self.model_iter_fn),
                                          ("dynamo", optimized_model_iter_fn)):
                    profile_fn(model, example_inputs)
                    synchronize()
                    profiler = NPUProfiler(
                        enable=True, level=f"level{self.args.enable_profiler}", warmup=10,
                        active=max(1, self.args.iterations - 10),
                        save_path=os.path.join(self.args.prof_output_path, name, phase),
                    )
                    profiler.start()
                    try:
                        for _ in range(self.args.iterations):
                            profile_fn(model, example_inputs)
                            synchronize()
                            profiler.step()
                    finally:
                        profiler.stop()
            compilation_time = dynamo_latency - eager_latency
            compression_ratio = (
                eager_peak_mem / dynamo_peak_mem if dynamo_peak_mem else 0.0
            )

            if experiment.func is speedup_experiment:
                experiment_kwargs["compilation_latency"] = compilation_time
                experiment_kwargs["compression_ratio"] = compression_ratio
                experiment_kwargs["eager_peak_mem"] = eager_peak_mem
                experiment_kwargs["dynamo_peak_mem"] = dynamo_peak_mem
                experiment_kwargs["dynamo_stats"] = dynamo_stats

            if not hasattr(model, name):
                model.name = name
            results.append(experiment(model, example_inputs, **experiment_kwargs))
            return " ".join(map(str, results))
