#ifndef FST_LIBRARY_H_
#define FST_LIBRARY_H_

#include <fst/fstlib.h>
#include <fst/script/fstscript.h>
#include <fst/script/script-impl.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <set>
#include <map>
#include <cstdlib>
#include <cstring>
// #include <chrono>


//DEFINE_int32(v, 10, "v");
const float mydelta = 1.0F/(8192.0F*4);

using namespace fst;
// using namespace std::chrono;

template <typename Arc>
void configure_kernel_compose_cache(ComposeFstOptions<Arc> &options) {
	const char *mode = std::getenv("MEDICC2_FST_CACHE_MODE");
	if (mode == nullptr || mode[0] == '\0' ||
			std::strcmp(mode, "default") == 0) {
		return;
	}

	const char *bytes = std::getenv("MEDICC2_FST_CACHE_BYTES");
	if (bytes != nullptr && bytes[0] != '\0') {
		char *end = nullptr;
		const unsigned long long parsed = std::strtoull(bytes, &end, 10);
		if (end != bytes && *end == '\0') {
			options.gc_limit = static_cast<size_t>(parsed);
			return;
		}
	}

	if (std::strcmp(mode, "streaming") == 0) {
		// A zero-byte limit makes OpenFST reuse its first cache state during
		// single-pass shortest-distance traversal.
		options.gc_limit = 0;
	}
}

//void shortest_path(script::FstClass &model, script::FstClass &input, script::FstClass &output, script::MutableFstClass* path, std::vector<script::WeightClass>* distance) {
void align_std_impl(script::FstClass &model, script::FstClass &input, script::FstClass &output, script::MutableFstClass* path) {	
	const Fst<StdArc> *tfst = model.GetFst<StdArc>();
	const Fst<StdArc> *ifst1 = input.GetFst<StdArc>();
	const Fst<StdArc> *ifst2 = output.GetFst<StdArc>();
	MutableFst<StdArc> *ofst = path->GetMutableFst<StdArc>();

	ArcSortFst<StdArc, OLabelCompare<StdArc> > input_sorted = ArcSortFst<StdArc, OLabelCompare<StdArc> >(*ifst1, OLabelCompare<StdArc>());
	ArcSortFst<StdArc, ILabelCompare<StdArc> > output_sorted = ArcSortFst<StdArc, ILabelCompare<StdArc> >(*ifst2, ILabelCompare<StdArc>());

	// set compose options
	ComposeFstOptions<StdArc> co;
	co.gc_limit=0;

	// Container for composition result.
	ComposeFst<StdArc> middle = ComposeFst<StdArc>(input_sorted, *tfst, co);
	ComposeFst<StdArc> result = ComposeFst<StdArc>(middle, output_sorted, co);

	std::vector<StdArc::Weight> typed_distance;
	//script::WeightClass distance;

	NaturalShortestFirstQueue<StdArc::StateId, StdArc::Weight> state_queue(typed_distance);
	ShortestPathOptions<StdArc, NaturalShortestFirstQueue<StdArc::StateId, StdArc::Weight>, AnyArcFilter<StdArc> > opts(&state_queue, AnyArcFilter<StdArc>(), 1, false, false, mydelta);
	opts.first_path=true;

	ShortestPath(result, ofst, &typed_distance, opts);
	//StdArc::StateId start = path->Start();

	//distance = script::WeightClass(typed_distance[start]);
	//return distance;
}

script::WeightClass score_std_impl(script::FstClass &model, script::FstClass &input, script::FstClass &output) {	
    const Fst<StdArc> *tfst = model.GetFst<StdArc>();
	const Fst<StdArc> *ifst1 = input.GetFst<StdArc>();
	const Fst<StdArc> *ifst2 = output.GetFst<StdArc>();

	// auto start = high_resolution_clock::now();	
	ArcSortFst<StdArc, OLabelCompare<StdArc> > input_sorted = ArcSortFst<StdArc, OLabelCompare<StdArc> >(*ifst1, OLabelCompare<StdArc>());
	ArcSortFst<StdArc, ILabelCompare<StdArc> > output_sorted = ArcSortFst<StdArc, ILabelCompare<StdArc> >(*ifst2, ILabelCompare<StdArc>());

	// set compose options
	ComposeFstOptions<StdArc> co;
	co.gc_limit=0;

	// Container for composition result.
	ComposeFst<StdArc> middle = ComposeFst<StdArc>(input_sorted, *tfst, co);
	ComposeFst<StdArc> result = ComposeFst<StdArc>(middle, output_sorted, co);

	std::vector<StdArc::Weight> typed_distance;
	StdArc::Weight retval;
	script::WeightClass distance;

	NaturalShortestFirstQueue<StdArc::StateId, StdArc::Weight> state_queue(typed_distance);
	ShortestDistanceOptions<StdArc, NaturalShortestFirstQueue<StdArc::StateId, StdArc::Weight>, AnyArcFilter<StdArc> > opts(&state_queue, AnyArcFilter<StdArc>());
	opts.first_path=true;
	
	using StateId = typename StdArc::StateId;
	using Weight = typename StdArc::Weight;

	if (Weight::Properties() & kRightSemiring) {
		ShortestDistance(result, &typed_distance, opts);
		if (typed_distance.size() == 1 && !typed_distance[0].Member()) {
			retval =  StdArc::Weight::NoWeight();
		}
		Adder<Weight> adder;  // maintains cumulative sum accurately
		for (StateId state = 0; state < typed_distance.size(); ++state) {
			adder.Add(Times(typed_distance[state], result.Final(state)));
		}
		retval = adder.Sum();
	} else {
		ShortestDistance(result, &typed_distance, true, mydelta);
		const auto state = result.Start();
		if (typed_distance.size() == 1 && !typed_distance[0].Member()) {
			retval = StdArc::Weight::NoWeight();
		}
		retval = state != kNoStateId && state < typed_distance.size() ? typed_distance[state] : Weight::Zero();
	}
	
	distance = script::WeightClass(retval);
    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<milliseconds>(stop - start);
    // std::cout << duration.count() << std::endl;
	return distance;
}

script::WeightClass score_log_impl(script::FstClass &model, script::FstClass &input, script::FstClass &output) {	
	const Fst<LogArc> *tfst = model.GetFst<LogArc>();
	const Fst<LogArc> *ifst1 = input.GetFst<LogArc>();
	const Fst<LogArc> *ifst2 = output.GetFst<LogArc>();
	
	ArcSortFst<LogArc, OLabelCompare<LogArc> > input_sorted = ArcSortFst<LogArc, OLabelCompare<LogArc> >(*ifst1, OLabelCompare<LogArc>());
	ArcSortFst<LogArc, ILabelCompare<LogArc> > output_sorted = ArcSortFst<LogArc, ILabelCompare<LogArc> >(*ifst2, ILabelCompare<LogArc>());

	// set compose options
	ComposeFstOptions<LogArc> co;
	//co.gc_limit=0;

	// Container for composition result.
	ComposeFst<LogArc> middle = ComposeFst<LogArc>(input_sorted, *tfst, co);
	ComposeFst<LogArc> result = ComposeFst<LogArc>(middle, output_sorted, co);

	std::vector<LogArc::Weight> typed_distance;
	LogArc::Weight retval;
	script::WeightClass distance;

	//NaturalShortestFirstQueue<LogArc::StateId, LogArc::Weight> state_queue(typed_distance);
	//ShortestDistanceOptions<LogArc, NaturalShortestFirstQueue<LogArc::StateId, LogArc::Weight>, AnyArcFilter<LogArc> > opts(&state_queue, AnyArcFilter<LogArc>());
	//opts.first_path=true;
	
	using StateId = typename LogArc::StateId;
	using Weight = typename LogArc::Weight;

	if (Weight::Properties() & kRightSemiring) {
		ShortestDistance(result, &typed_distance);
		if (typed_distance.size() == 1 && !typed_distance[0].Member()) {
			retval =  LogArc::Weight::NoWeight();
		}
		Adder<Weight> adder;  // maintains cumulative sum accurately
		for (StateId state = 0; state < typed_distance.size(); ++state) {
			adder.Add(Times(typed_distance[state], result.Final(state)));
		}
		retval = adder.Sum();
	} else {
		ShortestDistance(result, &typed_distance, true, mydelta);
		const auto state = result.Start();
		if (typed_distance.size() == 1 && !typed_distance[0].Member()) {
			retval = LogArc::Weight::NoWeight();
		}
		retval = state != kNoStateId && state < typed_distance.size() ? typed_distance[state] : Weight::Zero();
	}
	
	distance = script::WeightClass(retval);
	return distance;
}

script::WeightClass kernel_score_std_impl(script::FstClass &model, script::FstClass &input, script::FstClass &output) {	
	const Fst<StdArc> *tfst = model.GetFst<StdArc>();
	const Fst<StdArc> *ifst1 = input.GetFst<StdArc>();
	const Fst<StdArc> *ifst2 = output.GetFst<StdArc>();
	
	ArcSortFst<StdArc, ILabelCompare<StdArc> > input_sorted = ArcSortFst<StdArc, ILabelCompare<StdArc> >(*ifst1, ILabelCompare<StdArc>());
	ArcSortFst<StdArc, ILabelCompare<StdArc> > output_sorted = ArcSortFst<StdArc, ILabelCompare<StdArc> >(*ifst2, ILabelCompare<StdArc>());

	// set compose options
	ComposeFstOptions<StdArc> co;
	configure_kernel_compose_cache(co);

	// Container for composition result.
	InvertFst<StdArc> middle1 = InvertFst<StdArc>(ComposeFst<StdArc>(*tfst, input_sorted, co));
	ComposeFst<StdArc> middle2 = ComposeFst<StdArc>(*tfst, output_sorted, co);
	ArcSortFst<StdArc, ILabelCompare<StdArc> > middle2_sorted = ArcSortFst<StdArc, ILabelCompare<StdArc> >(middle2, ILabelCompare<StdArc>());
	//ArcSortFst<StdArc, OLabelCompare<StdArc> > middle1_sorted = ArcSortFst<StdArc, OLabelCompare<StdArc> >(middle1, OLabelCompare<StdArc>());
	ComposeFst<StdArc> result = ComposeFst<StdArc>(middle1, middle2_sorted, co);
	

	std::vector<StdArc::Weight> typed_distance;
	StdArc::Weight retval;
	script::WeightClass distance;

	NaturalShortestFirstQueue<StdArc::StateId, StdArc::Weight> state_queue(typed_distance);
	ShortestDistanceOptions<StdArc, NaturalShortestFirstQueue<StdArc::StateId, StdArc::Weight>, AnyArcFilter<StdArc> > opts(&state_queue, AnyArcFilter<StdArc>());
	opts.first_path=true;
	
	using StateId = typename StdArc::StateId;
	using Weight = typename StdArc::Weight;

	if (Weight::Properties() & kRightSemiring) {
		ShortestDistance(result, &typed_distance, opts);
		if (typed_distance.size() == 1 && !typed_distance[0].Member()) {
			retval =  StdArc::Weight::NoWeight();
		}
		Adder<Weight> adder;  // maintains cumulative sum accurately
		for (StateId state = 0; state < typed_distance.size(); ++state) {
			adder.Add(Times(typed_distance[state], result.Final(state)));
		}
		retval = adder.Sum();
	} else {
		ShortestDistance(result, &typed_distance, true, mydelta);
		const auto state = result.Start();
		if (typed_distance.size() == 1 && !typed_distance[0].Member()) {
			retval = StdArc::Weight::NoWeight();
		}
		retval = state != kNoStateId && state < typed_distance.size() ? typed_distance[state] : Weight::Zero();
	}
	
	distance = script::WeightClass(retval);
	return distance;
}

script::WeightClass kernel_score_log_impl(script::FstClass &model, script::FstClass &input, script::FstClass &output) {	
	const Fst<LogArc> *tfst = model.GetFst<LogArc>();
	const Fst<LogArc> *ifst1 = input.GetFst<LogArc>();
	const Fst<LogArc> *ifst2 = output.GetFst<LogArc>();
	
	ArcSortFst<LogArc, ILabelCompare<LogArc> > input_sorted = ArcSortFst<LogArc, ILabelCompare<LogArc> >(*ifst1, ILabelCompare<LogArc>());
	ArcSortFst<LogArc, ILabelCompare<LogArc> > output_sorted = ArcSortFst<LogArc, ILabelCompare<LogArc> >(*ifst2, ILabelCompare<LogArc>());

	// set compose options
	ComposeFstOptions<LogArc> co;
	configure_kernel_compose_cache(co);

	// Container for composition result.
	InvertFst<LogArc> middle1 = InvertFst<LogArc>(ComposeFst<LogArc>(*tfst, input_sorted, co));
	ComposeFst<LogArc> middle2 = ComposeFst<LogArc>(*tfst, output_sorted, co);
	ArcSortFst<LogArc, ILabelCompare<LogArc> > middle2_sorted = ArcSortFst<LogArc, ILabelCompare<LogArc> >(middle2, ILabelCompare<LogArc>());
	//ArcSortFst<LogArc, OLabelCompare<LogArc> > middle1_sorted = ArcSortFst<LogArc, OLabelCompare<LogArc> >(middle1, OLabelCompare<LogArc>());
	ComposeFst<LogArc> result = ComposeFst<LogArc>(middle1, middle2_sorted, co);
	

	std::vector<LogArc::Weight> typed_distance;
	LogArc::Weight retval;
	script::WeightClass distance;

	//NaturalShortestFirstQueue<LogArc::StateId, LogArc::Weight> state_queue(typed_distance);
	//ShortestDistanceOptions<LogArc, NaturalShortestFirstQueue<LogArc::StateId, LogArc::Weight>, AnyArcFilter<LogArc> > opts(&state_queue, AnyArcFilter<LogArc>());
	//opts.first_path=true;
	
	using StateId = typename LogArc::StateId;
	using Weight = typename LogArc::Weight;

	if (Weight::Properties() & kRightSemiring) {
		ShortestDistance(result, &typed_distance);
		if (typed_distance.size() == 1 && !typed_distance[0].Member()) {
			retval =  LogArc::Weight::NoWeight();
		}
		Adder<Weight> adder;  // maintains cumulative sum accurately
		for (StateId state = 0; state < typed_distance.size(); ++state) {
			adder.Add(Times(typed_distance[state], result.Final(state)));
		}
		retval = adder.Sum();
	} else {
		ShortestDistance(result, &typed_distance, true, mydelta);
		const auto state = result.Start();
		if (typed_distance.size() == 1 && !typed_distance[0].Member()) {
			retval = LogArc::Weight::NoWeight();
		}
		retval = state != kNoStateId && state < typed_distance.size() ? typed_distance[state] : Weight::Zero();
	}
	
	distance = script::WeightClass(retval);
	return distance;
}

script::WeightClass multi_score_std_impl(script::FstClass &loh, script::FstClass &wgd, script::FstClass &gl, script::FstClass &input, script::FstClass &output) {	
	const Fst<StdArc> *lohfst = loh.GetFst<StdArc>();
    const Fst<StdArc> *wgdfst = wgd.GetFst<StdArc>();
    const Fst<StdArc> *glfst = gl.GetFst<StdArc>();
	const Fst<StdArc> *ifst1 = input.GetFst<StdArc>();
	const Fst<StdArc> *ifst2 = output.GetFst<StdArc>();
	
	ArcSortFst<StdArc, ILabelCompare<StdArc> > ifst_sorted = ArcSortFst<StdArc, ILabelCompare<StdArc> >(*ifst1, ILabelCompare<StdArc>());
	ArcSortFst<StdArc, ILabelCompare<StdArc> > ofst_sorted = ArcSortFst<StdArc, ILabelCompare<StdArc> >(*ifst2, ILabelCompare<StdArc>());

	// set compose options
	ComposeFstOptions<StdArc> co;
	//co.gc_limit=0;

    ComposeFst<StdArc> result = ComposeFst<StdArc>(ComposeFst<StdArc>(ComposeFst<StdArc>(ComposeFst<StdArc>(ifst_sorted, *lohfst, co), *wgdfst, co), *glfst, co), ofst_sorted, co);

	std::vector<StdArc::Weight> typed_distance;
	StdArc::Weight retval;
	script::WeightClass distance;

	NaturalShortestFirstQueue<StdArc::StateId, StdArc::Weight> state_queue(typed_distance);
	ShortestDistanceOptions<StdArc, NaturalShortestFirstQueue<StdArc::StateId, StdArc::Weight>, AnyArcFilter<StdArc> > opts(&state_queue, AnyArcFilter<StdArc>());
	opts.first_path=true;
	
	using StateId = typename StdArc::StateId;
	using Weight = typename StdArc::Weight;

	if (Weight::Properties() & kRightSemiring) {
		ShortestDistance(result, &typed_distance, opts);
		if (typed_distance.size() == 1 && !typed_distance[0].Member()) {
			retval =  StdArc::Weight::NoWeight();
		}
		Adder<Weight> adder;  // maintains cumulative sum accurately
		for (StateId state = 0; state < typed_distance.size(); ++state) {
			adder.Add(Times(typed_distance[state], result.Final(state)));
		}
		retval = adder.Sum();
	} else {
		ShortestDistance(result, &typed_distance, true, mydelta);
		const auto state = result.Start();
		if (typed_distance.size() == 1 && !typed_distance[0].Member()) {
			retval = StdArc::Weight::NoWeight();
		}
		retval = state != kNoStateId && state < typed_distance.size() ? typed_distance[state] : Weight::Zero();
	}
	
	distance = script::WeightClass(retval);
	return distance;
}
script::WeightClass multi_kernel_score_std_impl(script::FstClass &loh, script::FstClass &wgd, script::FstClass &gain, script::FstClass &loss, script::FstClass &input, script::FstClass &output) {	
	const Fst<StdArc> *lohfst = loh.GetFst<StdArc>();
    const Fst<StdArc> *wgdfst = wgd.GetFst<StdArc>();
    const Fst<StdArc> *gainfst = gain.GetFst<StdArc>();
    const Fst<StdArc> *lossfst = loss.GetFst<StdArc>();
	const Fst<StdArc> *ifst1 = input.GetFst<StdArc>();
	const Fst<StdArc> *ifst2 = output.GetFst<StdArc>();
	
	ArcSortFst<StdArc, ILabelCompare<StdArc> > input_sorted = ArcSortFst<StdArc, ILabelCompare<StdArc> >(*ifst1, ILabelCompare<StdArc>());
	ArcSortFst<StdArc, ILabelCompare<StdArc> > output_sorted = ArcSortFst<StdArc, ILabelCompare<StdArc> >(*ifst2, ILabelCompare<StdArc>());

	// set compose options
	ComposeFstOptions<StdArc> co;
	configure_kernel_compose_cache(co);

    InvertFst<StdArc> left = InvertFst<StdArc>(ComposeFst<StdArc>(*lohfst, ComposeFst<StdArc>(*wgdfst, ComposeFst<StdArc>(*gainfst, ComposeFst<StdArc>(*lossfst, input_sorted, co), co), co), co));
    ComposeFst<StdArc> right = ComposeFst<StdArc>(*lohfst, ComposeFst<StdArc>(*wgdfst, ComposeFst<StdArc>(*gainfst, ComposeFst<StdArc>(*lossfst, output_sorted, co), co), co), co);
    ArcSortFst<StdArc, ILabelCompare<StdArc> > right_sorted = ArcSortFst<StdArc, ILabelCompare<StdArc> >(right, ILabelCompare<StdArc>());
	ComposeFst<StdArc> result = ComposeFst<StdArc>(left, right_sorted, co);

	std::vector<StdArc::Weight> typed_distance;
	StdArc::Weight retval;
	script::WeightClass distance;

	NaturalShortestFirstQueue<StdArc::StateId, StdArc::Weight> state_queue(typed_distance);
	ShortestDistanceOptions<StdArc, NaturalShortestFirstQueue<StdArc::StateId, StdArc::Weight>, AnyArcFilter<StdArc> > opts(&state_queue, AnyArcFilter<StdArc>());
	opts.first_path=true;
	
	using StateId = typename StdArc::StateId;
	using Weight = typename StdArc::Weight;

	if (Weight::Properties() & kRightSemiring) {
		ShortestDistance(result, &typed_distance, opts);
		if (typed_distance.size() == 1 && !typed_distance[0].Member()) {
			retval =  StdArc::Weight::NoWeight();
		}
		Adder<Weight> adder;  // maintains cumulative sum accurately
		for (StateId state = 0; state < typed_distance.size(); ++state) {
			adder.Add(Times(typed_distance[state], result.Final(state)));
		}
		retval = adder.Sum();
	} else {
		ShortestDistance(result, &typed_distance, true, mydelta);
		const auto state = result.Start();
		if (typed_distance.size() == 1 && !typed_distance[0].Member()) {
			retval = StdArc::Weight::NoWeight();
		}
		retval = state != kNoStateId && state < typed_distance.size() ? typed_distance[state] : Weight::Zero();
	}
	
	distance = script::WeightClass(retval);
	return distance;
}

#endif
