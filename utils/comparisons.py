def show_comparison(course1_data, course2_data):
    import streamlit as st
    st.subheader("🔍 Comparison by Aspect")

    aspect_keys = set(course1_data["aspect_sentiments"].keys()) | set(course2_data["aspect_sentiments"].keys())
    for aspect in sorted(aspect_keys):
        examples1 = course1_data["aspect_sentiments"].get(aspect, [])
        examples2 = course2_data["aspect_sentiments"].get(aspect, [])

        # Skip if both courses have no data for this aspect
        if not examples1 and not examples2:
            continue

        st.markdown(f"### 🔸 {aspect.replace('_', ' ').title()}")

        if examples1:
            st.markdown("**📘 Course 1 Examples:**")
            for sent, label in examples1[:3]:
                st.markdown(f"- {'⭐️' * (label + 1)} {sent}")
        else:
            st.markdown("_No examples found for Course 1._")

        if examples2:
            st.markdown("**📙 Course 2 Examples:**")
            for sent, label in examples2[:3]:
                st.markdown(f"- {'⭐️' * (label + 1)} {sent}")
        else:
            st.markdown("_No examples found for Course 2._")

        st.markdown("---")
