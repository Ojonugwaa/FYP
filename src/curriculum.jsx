import React, { useState } from "react";
//eslint-disable-next-line no-unused-vars
import { motion } from "framer-motion";
import { useNavigate } from "react-router";

function Curriculum() {
  const [jobTitle, setJobTitle] = useState("");
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const navigate = useNavigate();
  const gotoskills = () => {
    navigate("/");
  };
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResults(null);
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append("job_title", jobTitle);

      const response = await fetch("http://localhost:8000/analyze_curriculum", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error(await response.text());
      const data = await response.json();
      console.log("API Results:", data); // 🔍 Debug log
      setResults(data);
    } catch (err) {
      setError(err.message || "Failed to analyze curriculum.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 1 }}
      className="container mx-auto px-4 py-8 max-w-4xl"
    >
      <motion.div
        whileHover={{ scale: 1.01 }}
        className="bg-white rounded-lg shadow-md p-6 mb-8"
      >
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-gray-700 font-medium mb-2">
              Job Title
            </label>
            <input
              type="text"
              value={jobTitle}
              onChange={(e) => setJobTitle(e.target.value)}
              required
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              placeholder="e.g. Frontend Developer"
            />
          </div>

          <motion.button
            type="submit"
            whileTap={{ scale: 0.95 }}
            disabled={isLoading}
            className={`w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-4 rounded-lg transition duration-200 ${
              isLoading ? "opacity-70 cursor-not-allowed" : ""
            }`}
          >
            {isLoading ? "Analyzing..." : "Analyze Curriculum"}
          </motion.button>
          <div className="w-full flex justify-end">
            <button
              onClick={gotoskills}
              className="text-center bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-4 rounded-lg transition duration-200"
            >
              Skills Analyzer
            </button>
          </div>
        </form>

        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-6 p-4 bg-red-50 border-l-4 border-red-500 text-red-700"
          >
            <p>{error}</p>
          </motion.div>
        )}
      </motion.div>

      {results && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg shadow-md p-6"
        >
          <h3 className="text-xl font-bold text-gray-800 mb-4">
            Analysis Results
          </h3>

          <div className="text-gray-700 whitespace-pre-wrap">
            <p>
              <strong>Job Title:</strong> {results.job_title}
            </p>
            <p className="mt-4 mb-2">
              <strong>Relevant Courses:</strong>
            </p>

            {Array.isArray(results.relevant_courses) &&
            results.relevant_courses.length > 0 ? (
              <ul className="list-disc list-inside space-y-2">
                {results.relevant_courses.map((course, index) => (
                  <li key={index}>
                    <strong>{course.title}</strong>: {course.description}
                  </li>
                ))}
              </ul>
            ) : (
              <p>No relevant courses found.</p>
            )}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}

export default Curriculum;
