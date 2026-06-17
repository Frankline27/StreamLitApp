# Make prediction
predictions = model.predict(img_array, verbose=0)

# INTERPRET FIXED - Swap classes
if predictions.shape[-1] == 1:
    # The model outputs probability for class 0 (which is Non-MRI in your training)
    probability_non_mri = float(predictions[0][0])
    probability_mri = 1 - probability_non_mri
    
    # Swap the classification logic
    if probability_mri > 0.5:
        predicted_class = "MRI"
        confidence = probability_mri
    else:
        predicted_class = "Non-MRI"
        confidence = probability_non_mri
else:
    # For 2-class softmax, swap indices
    probability_mri = float(predictions[0][0])  # Assuming index 0 is MRI now
    probability_non_mri = float(predictions[0][1])
    
    if probability_mri > 0.5:
        predicted_class = "MRI"
        confidence = probability_mri
    else:
        predicted_class = "Non-MRI"
        confidence = probability_non_mri

# Display results
st.subheader("📊 Prediction Results")

col1, col2 = st.columns(2)
with col1:
    st.metric("Prediction", predicted_class)
with col2:
    st.metric("Confidence", f"{confidence:.2%}")

# Progress bar - NOW CORRECT
st.write("### Probability Distribution")
st.progress(float(probability_mri))
st.caption(f"MRI probability: {probability_mri:.2%} | Non-MRI probability: {probability_non_mri:.2%}")

if predicted_class == "MRI":
    st.info("🧠 This image appears to be an MRI scan.")
else:
    st.info("📷 This image does not appear to be an MRI scan.")
