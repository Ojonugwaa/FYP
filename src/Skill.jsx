// import React, { useState } from "react";
// // eslint-disable-next-line no-unused-vars
// import { motion } from "framer-motion";
// import { FiUpload, FiCheckCircle } from "react-icons/fi"; // Icons for upload

// function Skill() {
//   const [jobTitle, setJobTitle] = useState("");
//   const [cvSkills, setCvSkills] = useState("");
//   const [results, setResults] = useState(null);
//   const [error, setError] = useState("");
//   const [file, setFile] = useState(null);
//   const [isLoading, setIsLoading] = useState(false);
//   const [uploadSuccess, setUploadSuccess] = useState(false);

//   const handleFileChange = (e) => {
//     const selectedFile = e.target.files[0];
//     if (
//       selectedFile &&
//       (selectedFile.type === "application/pdf" ||
//         selectedFile.type.includes("word"))
//     ) {
//       setFile(selectedFile);
//       setUploadSuccess(true);
//       setTimeout(() => setUploadSuccess(false), 3000); // Reset after 3s

//       // You can add your logic here to extract job title and skills from the uploaded CV.
//       // For now, I'm just setting mock data.
//       setJobTitle("Frontend Developer");
//       setCvSkills("React, JavaScript, CSS, HTML");
//     } else {
//       setError("Please upload a PDF or DOCX file.");
//     }
//   };

//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     setError("");
//     setResults(null);
//     setIsLoading(true);

//     try {
//       const formData = new FormData();
//       formData.append("job_title", jobTitle);

//       // Ensure cvSkills is split into an array, even if only one skill is provided
//       const cvSkillsArray = cvSkills.split(",").map((skill) => skill.trim());
//       formData.append("cv_skills", cvSkillsArray);

//       if (file) formData.append("cv_file", file);

//       const response = await fetch("http://localhost:8000/analyze_skills", {
//         method: "POST",
//         body: formData,
//       });

//       if (!response.ok) throw new Error(await response.text());
//       const data = await response.json();
//       setResults(data);
//     } catch (err) {
//       setError(err.message || "Failed to analyze skills.");
//     } finally {
//       setIsLoading(false);
//     }
//   };

//   return (
//     <motion.div
//       initial={{ opacity: 0 }}
//       animate={{ opacity: 1 }}
//       transition={{ duration: 1 }}
//       className="container mx-auto px-4 py-8 max-w-4xl"
//     >
//       <motion.div
//         whileHover={{ scale: 1.01 }}
//         className="bg-white rounded-lg shadow-md p-6 mb-8"
//       >
//         <form onSubmit={handleSubmit} className="space-y-4">
//           {/* Job Title Input - Only visible if no CV is uploaded */}
//           {!file && (
//             <div>
//               <label className="block text-gray-700 font-medium mb-2">
//                 Job Title
//               </label>
//               <input
//                 type="text"
//                 value={jobTitle}
//                 onChange={(e) => setJobTitle(e.target.value)}
//                 required
//                 className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
//                 placeholder="e.g. Frontend Developer"
//               />
//             </div>
//           )}

//           {/* Skills Input - Only visible if no CV is uploaded */}
//           {!file && (
//             <div>
//               <label className="block text-gray-700 font-medium mb-2">
//                 Your Skills (comma-separated)
//               </label>
//               <textarea
//                 rows="4"
//                 value={cvSkills}
//                 onChange={(e) => setCvSkills(e.target.value)}
//                 required
//                 className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
//                 placeholder="e.g. React, JavaScript, CSS, HTML"
//               />
//             </div>
//           )}

//           {/* File Upload Section */}
//           <div className="space-y-2">
//             <label className="block text-gray-700 font-medium mb-2">
//               Upload CV (Optional)
//             </label>
//             <div className="flex items-center gap-4">
//               <label className="flex-1 cursor-pointer">
//                 <div
//                   className={`flex items-center justify-center px-4 py-2 border-2 border-dashed rounded-lg ${
//                     uploadSuccess
//                       ? "border-green-500 bg-green-50"
//                       : "border-gray-300 hover:border-indigo-500"
//                   }`}
//                 >
//                   <div className="flex items-center gap-2">
//                     <FiUpload className="text-gray-500" />
//                     <span className="text-gray-600">
//                       {file ? file.name : "Click to upload PDF/DOCX"}
//                     </span>
//                   </div>
//                   <input
//                     type="file"
//                     onChange={handleFileChange}
//                     accept=".pdf,.docx"
//                     className="hidden"
//                   />
//                 </div>
//               </label>
//               {uploadSuccess && (
//                 <motion.div
//                   initial={{ scale: 0 }}
//                   animate={{ scale: 1 }}
//                   className="text-green-600 flex items-center gap-1"
//                 >
//                   <FiCheckCircle />
//                   <span>Uploaded!</span>
//                 </motion.div>
//               )}
//             </div>
//           </div>

//           <motion.button
//             type="submit"
//             whileTap={{ scale: 0.95 }}
//             disabled={isLoading}
//             className={`w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-4 rounded-lg transition duration-200 ${
//               isLoading ? "opacity-70 cursor-not-allowed" : ""
//             }`}

//           >
//             {isLoading ? "Analyzing..." : "Analyze Skills"}

//           </motion.button>
//         </form>

//         {error && (
//           <motion.div
//             initial={{ opacity: 0, y: -10 }}
//             animate={{ opacity: 1, y: 0 }}
//             className="mt-6 p-4 bg-red-50 border-l-4 border-red-500 text-red-700"
//           >
//             <p>{error}</p>
//           </motion.div>
//         )}
//       </motion.div>

//       {results && (
//         <motion.div
//           initial={{ opacity: 0, y: 20 }}
//           animate={{ opacity: 1, y: 0 }}
//           className="bg-white rounded-lg shadow-md p-6"
//         >
//           <h3 className="text-xl font-semibold text-gray-800 mb-4">
//             Results for: {results.job_title}
//           </h3>

//           <p className="mb-4">
//             <strong>Missing Skills:</strong>{" "}
//             {results.missing_skills.length > 0
//               ? results.missing_skills.join(", ")
//               : "None 🎉"}
//           </p>

//           <h4 className="text-lg font-medium mb-2">Course Recommendations</h4>
//           {Object.entries(results.course_recommendations).map(
//             ([skill, platforms]) => (
//               <div key={skill} className="mb-4">
//                 <h5 className="font-semibold">{skill}</h5>
//                 <ul className="list-disc ml-6">
//                   {platforms.coursera.map((course, idx) => (
//                     <li key={`coursera-${idx}`}>
//                       Coursera:{" "}
//                       <a
//                         href={course.course_url}
//                         target="_blank"
//                         rel="noopener noreferrer"
//                         className="text-indigo-600 hover:underline"
//                       >
//                         {course.Title}
//                       </a>
//                     </li>
//                   ))}
//                   {platforms.udemy.map((course, idx) => (
//                     <li key={`udemy-${idx}`}>
//                       Udemy:{" "}
//                       <a
//                         href={course.course_url}
//                         target="_blank"
//                         rel="noopener noreferrer"
//                         className="text-indigo-600 hover:underline"
//                       >
//                         {course.Title}
//                       </a>
//                     </li>
//                   ))}
//                 </ul>
//               </div>
//             )
//           )}
//         </motion.div>
//       )}
//     </motion.div>
//   );
// }

// export default Skill;

import React, { useState } from "react";
//eslint-disable-next-line no-unused-vars
import { motion } from "framer-motion";
import { FiUpload, FiCheckCircle } from "react-icons/fi";
import { Link, Navigate } from "react-router";
import { useNavigate } from "react-router";

function Skill() {
  const [jobTitle, setJobTitle] = useState("");
  const [cvSkills, setCvSkills] = useState("");
  const [file, setFile] = useState(null);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const navigate = useNavigate()
  const gotocurriculum = () => {
    navigate('/curriculum');
  }
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (
      selectedFile &&
      (selectedFile.type === "application/pdf" ||
        selectedFile.type.includes("word"))
    ) {
      setFile(selectedFile);
      setUploadSuccess(true);
      setTimeout(() => setUploadSuccess(false), 3000);
      // ⚡ Do NOT overwrite job title or skills — let backend extract
    } else {
      setError("Please upload a PDF or DOCX file.");
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResults(null);
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append("job_title", jobTitle);
      if (cvSkills.trim()) {
        const skillsArray = cvSkills.split(",").map((skill) => skill.trim());
        for (let skill of skillsArray) {
          formData.append("cv_skills", skill);
        }
      }
      if (file) {
        formData.append("cv_file", file);
      }

      const response = await fetch("http://localhost:8000/analyze_skills", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error(await response.text());
      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message || "Failed to analyze skills.");
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
          {/* Upload Section */}
          <div className="space-y-2">
            <label className="block text-gray-700 font-medium mb-2">
              Upload CV (Optional)
            </label>
            <div className="flex items-center gap-4">
              <label className="flex-1 cursor-pointer">
                <div
                  className={`flex items-center justify-center px-4 py-2 border-2 border-dashed rounded-lg ${
                    uploadSuccess
                      ? "border-green-500 bg-green-50"
                      : "border-gray-300 hover:border-indigo-500"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <FiUpload className="text-gray-500" />
                    <span className="text-gray-600">
                      {file ? file.name : "Click to upload PDF/DOCX"}
                    </span>
                  </div>
                  <input
                    type="file"
                    onChange={handleFileChange}
                    accept=".pdf,.docx"
                    className="hidden"
                  />
                </div>
              </label>
              {uploadSuccess && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="text-green-600 flex items-center gap-1"
                >
                  <FiCheckCircle />
                  <span>Uploaded!</span>
                </motion.div>
              )}
            </div>
          </div>

          {/* Manual Job Title and Skills Input — Only show if NO file */}
          {!file && (
            <>
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
              <div>
                <label className="block text-gray-700 font-medium mb-2">
                  Your Skills (comma-separated)
                </label>
                <textarea
                  rows="4"
                  value={cvSkills}
                  onChange={(e) => setCvSkills(e.target.value)}
                  required
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                  placeholder="e.g. React, JavaScript, CSS, HTML"
                />
              </div>
            </>
          )}

          <motion.button
            type="submit"
            whileTap={{ scale: 0.95 }}
            disabled={isLoading}
            className={`w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-4 rounded-lg transition duration-200 ${
              isLoading ? "opacity-70 cursor-not-allowed" : ""
            }`}
          >
            {isLoading ? "Analyzing..." : "Analyze Skills"}
          </motion.button>

          <div className="w-full flex justify-end">
          <button onClick={gotocurriculum} className="text-center bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-4 rounded-lg transition duration-200">
            Curriculum Analyzer
          </button>
          </div>
        </form>

        {/* Error Handling */}
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

      {/* Results Display */}
      {results && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg shadow-md p-6"
        >
          <h3 className="text-xl font-bold text-gray-800 mb-4">
            Results for: {results.job_title}
          </h3>
          <p className="mb-4">
            <strong>Missing Skills:</strong>{" "}
            {results.missing_skills.length > 0
              ? results.missing_skills.join(", ")
              : "None 🎉"}
          </p>

          <h4 className="text-lg font-medium mb-2">Course Recommendations</h4>
          {Object.entries(results.course_recommendations).map(
            ([skill, platforms]) => (
              <div key={skill} className="mb-4">
                <h5 className="font-semibold">{skill}</h5>
                <ul className="list-disc ml-6">
                  {platforms.coursera.map((course, idx) => (
                    <li key={`coursera-${idx}`}>
                      Coursera:{" "}
                      <a
                        href={course.course_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-indigo-600 hover:underline"
                      >
                        {course.Title}
                      </a>
                    </li>
                  ))}
                  {platforms.udemy.map((course, idx) => (
                    <li key={`udemy-${idx}`}>
                      Udemy:{" "}
                      <a
                        href={course.course_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-indigo-600 hover:underline"
                      >
                        {course.Title}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )
          )}
        </motion.div>
      )}
    </motion.div>
  );
}

export default Skill;
